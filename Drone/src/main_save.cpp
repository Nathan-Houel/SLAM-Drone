///////////////////////////////////////////////////////////
// TELLOVS: Visual servoing with Ryze Tello
// by jacques.gangloff@unistra.fr, Feb 2024
// [Tello API from https://github.com/HerrNamenlos123/tello]
//
// Set tab space = 2 in your editor for best viewing
//
// To compile:
//		mkdir build
//		cd build
//		cmake ..
//		make
//
// To test image processing:
//		Comment flag TELLO_FLIGHT below
//		Modify TELLO_SSID below according to your drone
//		Compile
//		Connect your WiFi to your Tello SSID
//		./tellovs
//		Wait for the live stream to appear
//		Bring a target into the field of view
//		(The target should be in portrait orientation)
//		Verify that the blobs are detected
//		Verify the validity of the velocity screw coordinates
//
// To fly:
//		Uncomment flag TELLO_FLIGHT below
//		Read the comment after the keyword [QUESTION1] and code the
//		image Jacobian computation
//		Read the comment after the keyword [QUESTION2] and code the
//		control signal allocation
//		Compile and modify your code until there is no error and no warning
//		./tellovs
//		Verify that manual control works (see [QUESTION3] below)
//		Bring a target into the field of view
//		(The target should be in portrait orientation)
//		If the tracking does not work, debug your code
///////////////////////////////////////////////////////////

#include <math.h>
#include <pthread.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

#include "tello.hpp"

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/videoio.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/core/utils/logger.hpp"

using namespace std;

// Uncomment the line below to force a reboot at each run
#define		TELLO_REBOOT

// Uncomment the line below to activate flight
#define		TELLO_FLIGHT

// Modify TELLO_SSID according to your drone number
#define		TELLO_SSID							"TELLO-EE055A"
#define		TELLO_NB_DOF						4
#define		TELLO_LOOP_PERIOD_US		30000
#define		TELLO_MAX_CHAR					255
#define		TELLO_TXT_MARGIN				20
#define		TELLO_TXT_SIZE					0.5
#define		TELLO_TXT_THICKNESS			1
#define		TELLO_FEATURE_RADIUS		2
#define		TELLO_WAIT_AFTER_REBOOT	15
#define		TELLO_WINDOW_NAME				"Tello stream"
#define		TELLO_CAP_BUFF_SZ				1
#define		TELLO_CAMERA_WIDTH			648
#define		TELLO_CAMERA_HEIGHT			478
#define		TELLO_CAMERA_GX					620
#define		TELLO_CAMERA_GY					620
#define		TELLO_CAMERA_U0					324
#define		TELLO_CAMERA_V0					239
#define		TELLO_CAMERA_Z					0.3
#define		TELLO_NB_BLOBS					4
#define		TELLO_MIN_BLOB_SIZE_LOW	600
#define		TELLO_MIN_BLOB_SIZE_HI	1200
#define		TELLO_VS_SAT						30
#define		TELLO_MANUAL_V					TELLO_VS_SAT

#define		TELLO_KEY_UP						82
#define		TELLO_KEY_DOWN					84
#define		TELLO_KEY_RIGHT					83
#define		TELLO_KEY_LEFT					81
#define		TELLO_KEY_ESC						27

// Visual servoing tuning gains
#define		TELLO_VS_GAIN_LR				300
#define		TELLO_VS_GAIN_FB				250
#define		TELLO_VS_GAIN_UD				600
#define		TELLO_VS_GAIN_YAW				300

// Globals
pthread_t 												frame_thread;
pthread_mutex_t 									frame_mutex;
cv::Mat 													frame;
unsigned long long								frame_fps = 0;
bool															tello_exit = false;

// Fonction pour vérifier si le répertoire existe
bool directory_exists(const std::string& path) {
    struct stat info;
    return (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));
}

// Fonction pour créer le répertoire s'il n'existe pas
void create_directory(const std::string& path) {
    if (mkdir(path.c_str(), 0777) == -1 && errno != EEXIST) {
        std::cerr << "Error creating directory: " << path << std::endl;
    }
}

// Fonction pour générer le nom de fichier avec horodatage
std::string get_timestamp_filename() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm local_time = *std::localtime(&now_time_t);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y%m%d_%H%M%S_") << std::setw(3) << std::setfill('0') << millis;
    return oss.str();
}

// Fonction pour enregistrer l'image avec un chemin personnalisé
void save_image_with_timestamp(const cv::Mat &image, const std::string &directory) {
    // Vérifier si le répertoire existe, sinon le créer
    if (!directory_exists(directory)) {
        create_directory(directory);
    }

    // Générer le nom de fichier avec l'horodatage
    std::string filename = directory + "/image_" + get_timestamp_filename() + ".jpg";

    // Sauvegarder l'image
    cv::imwrite(filename, image);
    std::cout << "[Tello] Image saved: " << filename << std::endl;
}

//
//	Frame grabbing thread
//
void *tello_frame( void *ptr )	{
	struct timespec ts_tic, ts_tac;
	cv::Mat 				l_frame;
	
	pthread_mutex_lock( &frame_mutex );
	fprintf( stderr, "[Tello] Open a video stream socket..." );
	cv::VideoCapture capture{"udp://0.0.0.0:11111", cv::CAP_FFMPEG};
	capture.set(cv::CAP_PROP_BUFFERSIZE, TELLO_CAP_BUFF_SZ);
	fprintf( stderr, "done\n" );
	pthread_mutex_unlock( &frame_mutex );
	
	while( !tello_exit )	{
		// Grab frame
		clock_gettime( CLOCK_MONOTONIC, &ts_tic );
		capture >> l_frame;
		pthread_mutex_lock( &frame_mutex );
		l_frame.copyTo( frame );
		pthread_mutex_unlock( &frame_mutex );
		clock_gettime( CLOCK_MONOTONIC, &ts_tac );
		
		// Compute FPS
		frame_fps = (unsigned long long)ts_tac.tv_sec * 1000000000
									+ (unsigned long long)ts_tac.tv_nsec -
								( (unsigned long long)ts_tic.tv_sec * 1000000000
									+ (unsigned long long)ts_tic.tv_nsec );
		frame_fps = 1000000000 / frame_fps;
	}
	
	pthread_mutex_lock( &frame_mutex );
	fprintf( stderr, "[Tello] Closing video stream socket..." );
	capture.release();
	fprintf( stderr, "done\n" );
	pthread_mutex_unlock( &frame_mutex );
	
	return NULL;
}

//
//	Main
//
int main() {
    char	buf[TELLO_MAX_CHAR];
    Tello tello;
    float tello_control[TELLO_NB_DOF] = { 0.0 };
    std::string save_directory = "../images";
    
    // Mutex init
    pthread_mutex_init( &frame_mutex, NULL );
    
    printf( "[Tello] Visual servoing. JG, Feb 2025\n" );
    
    // Connect to Tello
    printf( "[Tello] Trying to connect to Tello...\n" );
    if (!tello.connect()) return 0;
    
    // Reboot Tello to improve reliability
    #ifdef TELLO_REBOOT
    printf( "[Tello] Rebooting Tello...\n" );
    tello.reboot();
    for ( int i = 0; i < TELLO_WAIT_AFTER_REBOOT; i++ )	{
			fprintf( stderr, "\33[2K\r" );
			fprintf( stderr, "[Tello] %d%% reboot completed", ( (i+1) * 100 ) / TELLO_WAIT_AFTER_REBOOT );
			sleep( 1 );
		}
		printf( "\n" );
    printf( "[Tello] Tello successfully rebooted\n" );
    fprintf( stderr, "[Tello] Reconnecting to %s...", TELLO_SSID );
    snprintf( buf, TELLO_MAX_CHAR, "nmcli connection up %s > /dev/null", TELLO_SSID );
    system( buf );
    printf( "done\n" );
    if (!tello.connect()) return 0;
    #endif
    
    // Set front camera to low resolution and start streaming
    tello.execute_manual_command( "setresolution low", TELLO_DEFAULT_COMMAND_TIMEOUT );
    printf( "[Tello] Enabling video streaming..." );
    tello.enable_video_stream();
    printf( "done\n" );
		
		// Start frame the grabbing thread
    pthread_create( &frame_thread, NULL, tello_frame, (void*) NULL );
    
    // Wait for grabbing thread initialization to complete
    usleep( TELLO_LOOP_PERIOD_US );
    pthread_mutex_lock( &frame_mutex );
    pthread_mutex_unlock( &frame_mutex );
    
    // Main loop for image processing and control
    int frame_count = 0;
    const int frames_per_second = 2; // 2 photos per second (500ms)
    const int interval_ms = 1000 / frames_per_second; // 500ms interval
    
    while (true) {
        pthread_mutex_lock( &frame_mutex );
        cv::Mat l_frame = frame;
        pthread_mutex_unlock( &frame_mutex );
        
        // Take a picture every 500ms (2 pictures per second)
		if (!l_frame.empty()) {
			save_image_with_timestamp(l_frame, save_directory);
		} else {
			std::cerr << "[Tello] Aucune image capturée!" << std::endl;
		}
        
        frame_count++;
        
        // Wait for the next frame (500ms)
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    // Stop the video stream and cleanup
    tello.disable_video_stream();
    pthread_join( frame_thread, NULL );
    pthread_mutex_destroy( &frame_mutex );
    
    return 0;
}
