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

#include "tello.hpp"

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/features2d.hpp"
#include "opencv4/opencv2/videoio.hpp"
#include "opencv4/opencv2/imgproc.hpp"
#include "opencv4/opencv2/core/utils/logger.hpp"

using namespace std;

// Uncomment the line below to force a reboot at each run
#define		TELLO_REBOOT

// Uncomment the line below to activate flight
#define		TELLO_FLIGHT

// Modify TELLO_SSID according to your drone number
//	01:
//	02:
//	03:
//	04:
//	05: EE0518
//	06: EE06E3
//	07:
//	08:
//	09:
//	10:
//	11:
//	12: EE057A
//	13: EE0678
//	14:	EE067A
//	15:	EE0832
//	16: EE0677
//	17:	EE05FB
//	18: EE05B4
//	19:	EE05A7
//	20:
//	21: EE07FC
//	22:	EE055A
//	23: EE06F0 
//	24:
//  25: EE077E
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
//#define	TELLO_CAMERA_WIDTH			960
//#define	TELLO_CAMERA_HEIGHT			720
#define		TELLO_CAMERA_WIDTH			648
#define		TELLO_CAMERA_HEIGHT			478
//#define	TELLO_CAMERA_GX					920
//#define	TELLO_CAMERA_GY					920
//#define	TELLO_CAMERA_U0					480
//#define	TELLO_CAMERA_V0					360
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

// Visual servoing reference primitives
double 														x_r[TELLO_NB_BLOBS] = { -50.0, 	50.0,		50.0,	-50.0 };
double														y_r[TELLO_NB_BLOBS] = { -106.0, -106.0, 106.0, 106.0 };

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
    
    // Takeoff
    #ifdef TELLO_FLIGHT
    printf( "[Tello] Drone takeoff..." );
    tello.takeoff();
    printf( "done\n" );
    #endif
    
    // Setup SimpleBlobDetector parameters
		cv::SimpleBlobDetector::Params params;
		 
		// Change thresholds
		//params.minThreshold = 100;
		//params.maxThreshold = 255;
		
		// Filter by color
		params.filterByColor = false;
		//params.blobColor = 150;
		 
		// Filter by Area.
		params.filterByArea = true;
		params.minArea = TELLO_MIN_BLOB_SIZE_HI;
		params.maxArea = 50000;
		 
		// Filter by Circularity
		params.filterByCircularity = true;
		//params.minCircularity = 0.5;
		 
		// Filter by Convexity
		//params.filterByConvexity = true;
		//params.minConvexity = 0.87;
		 
		// Filter by Inertia
		//params.filterByInertia = true;
		//params.minInertiaRatio = 0.01;
		
		// Initialize variables before entering the main loop
		cv::Mat 												l_frame;
		cv::Point 											textOrg( TELLO_TXT_MARGIN, TELLO_TXT_MARGIN );
		bool 														tello_servo = false;
		int															Key_pressed = 0;
		double 													x[TELLO_NB_BLOBS], 
																		y[TELLO_NB_BLOBS];
		
		// Main loop
    while (true) {
			
				// Check for an empty frame, if not, copy current frame
				pthread_mutex_lock( &frame_mutex );
        if ( frame.empty( ) )	{
					pthread_mutex_unlock( &frame_mutex );
					continue;
				}
				else {
					frame.copyTo( l_frame );
					pthread_mutex_unlock( &frame_mutex );
				}
				
				// Check frame dimensions
				if ( ( l_frame.cols != TELLO_CAMERA_WIDTH ) ||
						 ( l_frame.rows != TELLO_CAMERA_HEIGHT ) )	{
					printf( "\n" );
					printf( "[Tello] Unexpected frame dimensions, width: %d\tHeight: %d\n", l_frame.cols, l_frame.rows  );
					break;
				}
				
				// Detect blobs in frame
				std::vector<cv::KeyPoint> keypoints;
				cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
				detector->detect( l_frame, keypoints);
				
				// Draw detected blobs
				drawKeypoints( l_frame, keypoints, l_frame, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

				// Check if all blobs are detected
				if ( keypoints.size() == TELLO_NB_BLOBS )	{
					
					// Sorting keypoints according to its angle with respect to the center
					// 1 | 2
					// -----
					// 4 | 3
					double kp_angle[TELLO_NB_BLOBS];
					double kp_x[TELLO_NB_BLOBS];
					double kp_y[TELLO_NB_BLOBS];
					double kp_cog_x = 0.0;
					double kp_cog_y = 0.0;
					for ( int i = 0; i < TELLO_NB_BLOBS; i++ )	{
						
						// Compute feature coordinates in image-centered camera frame
						kp_x[i] = keypoints[i].pt.x - TELLO_CAMERA_U0;
						kp_y[i] = keypoints[i].pt.y - TELLO_CAMERA_V0;
						
						// Compute center of gravity
						kp_cog_x += kp_x[i];
						kp_cog_y += kp_y[i];
					}
					
					kp_cog_x /= TELLO_NB_BLOBS;
					kp_cog_y /= TELLO_NB_BLOBS;
					
					// Compute angle
					for ( int i = 0; i < TELLO_NB_BLOBS; i++ )	{	
						kp_angle[i] = atan2( kp_y[i] - kp_cog_y, kp_x[i] - kp_cog_x );
					}
					
					// Sort keypoints by descending order of angle
					for ( int i = 0 ; i < TELLO_NB_BLOBS-1; i++ )	{
						for( int j = 0 ; j < TELLO_NB_BLOBS-i-1; j++ )	{
							if( kp_angle[j] > kp_angle[j+1] ) {
								double swap_a = kp_angle[j];
								double swap_x = kp_x[j];
								double swap_y = kp_y[j];
								kp_angle[j] = kp_angle[j+1];
								kp_x[j] = kp_x[j+1];
								kp_y[j] = kp_y[j+1];
								kp_angle[j+1] = swap_a;
								kp_x[j+1] = swap_x;
								kp_y[j+1] = swap_y;
							}
						}
					}
					
					// If blobs were previously missing, store features as reference
					if ( !tello_servo )	{
						tello_servo = true;
						params.minArea = TELLO_MIN_BLOB_SIZE_LOW;
					}
					
					// Draw feature points
					for ( int i = 0; i < TELLO_NB_BLOBS; i++ )	{	
						circle( l_frame, 
										cv::Point( kp_x[i] + TELLO_CAMERA_U0, kp_y[i] + TELLO_CAMERA_V0 ), 
										TELLO_FEATURE_RADIUS, 
										cv::Scalar(0, 0, 255), 
										cv::FILLED, 
										cv::LINE_AA );
						snprintf( buf, TELLO_MAX_CHAR, "M%d", i );
						putText( 	l_frame, 
											buf, 
											cv::Point( kp_x[i] + TELLO_CAMERA_U0, kp_y[i] + TELLO_CAMERA_V0 + TELLO_TXT_MARGIN ), 
											cv::FONT_HERSHEY_SIMPLEX, 
											TELLO_TXT_SIZE, 
											cv::Scalar(0, 0, 255), 
											TELLO_TXT_THICKNESS, 
											cv::LINE_AA );
						circle( l_frame, 
										cv::Point( x_r[i] + TELLO_CAMERA_U0, y_r[i] + TELLO_CAMERA_V0 ), 
										TELLO_FEATURE_RADIUS, 
										cv::Scalar(0, 255, 0), 
										cv::FILLED, 
										cv::LINE_AA );
						snprintf( buf, TELLO_MAX_CHAR, "R%d", i );
						putText( 	l_frame, 
											buf, 
											cv::Point( x_r[i] + TELLO_CAMERA_U0, y_r[i] + TELLO_CAMERA_V0 - TELLO_TXT_MARGIN ), 
											cv::FONT_HERSHEY_SIMPLEX, 
											TELLO_TXT_SIZE, 
											cv::Scalar(0, 255, 0), 
											TELLO_TXT_THICKNESS, 
											cv::LINE_AA );
					}
					
					// Define image Jacobian matrix
					cv::Mat Lf = cv::Mat::zeros( 2 * TELLO_NB_BLOBS, TELLO_NB_DOF, CV_64F );
					
					// [QUESTION1] Compute image Jacobian
					// Notes: 
					//		- Define a specific element of matrix Lf by 
					// 			Lf.at<double>(line,col)=val, line=0...7, col=0...5
					// 		-	Camera intrinsic parameters Gx, Gy are defined by macros
					// 			TELLO_CAMERA_GX and TELLO_CAMERA_GY
					// 		-	The depth of all points is supposed constant equal to
					// 			TELLO_CAMERA_Z
					//		-	The coordinates in pixels of the center of each TELLO_NB_BLOBS 
					//			features are stored in kp_x[i] and kp_y[i], i=0...3
					//		- The camera frame has its origin at image center
					//		- Camera frame axis:
					//			---------------------------
					//			|													|
					//			|													|
					//			|						-------> X		|
					//			|						|							|
					//			|						|							|
					//			|						V Y						|
					//			---------------------------
					
					
					for ( int i = 0; i < TELLO_NB_BLOBS; i++ )	{
						Lf.at<double>(2*i,0) = - TELLO_CAMERA_GX / TELLO_CAMERA_Z;
						Lf.at<double>(2*i,1) = 0.0;
						Lf.at<double>(2*i,2) = kp_x[i] / TELLO_CAMERA_Z;
						Lf.at<double>(2*i,3) = -( TELLO_CAMERA_GX * TELLO_CAMERA_GX + kp_x[i] * kp_x[i] ) / TELLO_CAMERA_GX;
						
						Lf.at<double>(2*i+1,0) = 0.0;
						Lf.at<double>(2*i+1,1) = -TELLO_CAMERA_GY / TELLO_CAMERA_Z;
						Lf.at<double>(2*i+1,2) = kp_y[i] / TELLO_CAMERA_Z;
						Lf.at<double>(2*i+1,3) = - kp_x[i] * kp_y[i] / TELLO_CAMERA_GX;
					}
					
					// Pseudo-inverse of image Jacobian
					cv::Mat LftLf = Lf.t() * Lf;
					cv::Mat iLftLf = LftLf.inv();
					cv::Mat piLf = iLftLf * Lf.t();
					
					// Compute the error vector
					// Note: 	-	the reference features vectors x_r and y_r are
					//					defined at the top of this file
					cv::Mat err = cv::Mat::zeros( 2*TELLO_NB_BLOBS, 1, CV_64F );
					for ( int i = 0; i < TELLO_NB_BLOBS; i++ )	{
						err.at<double>(2*i,0) = x_r[i] - kp_x[i];
						err.at<double>(2*i+1,0) = y_r[i] - kp_y[i];
					}
					
					// Compute the velocity screw
					cv::Mat Cc = piLf * err;
					
					// [QUESTION2] Define the control signals
					// Notes:
					//		-	The ith coordinate of the velocity screw Cc can be
					//			accessed by Cc.at<double>(i,0)
					//		- The gains of the Left-Right, Front-Back, Up-Down and
					//			Yaw control inputs of the drone can be tuned separately
					//			by the macros TELLO_VS_GAIN_LR, TELLO_VS_GAIN_FB, 
					//			TELLO_VS_GAIN_UD, TELLO_VS_GAIN_YAW respectively
					//			(see at the begin of this file at the end of the defines)
					//		- The Left-Right, Front-Back, Up-Down and
					//			Yaw control inputs of the drone are defined in
					//			tello_control[0] ... tello_control[3] respectively with the
					// 			following positive direction convention:
					//				LR+  : right
					// 				FB+  : front
					// 				UD+  : up
					// 				Yaw+ : clockwise when looking from the top
					
					// Left-Right
					tello_control[0] = TELLO_VS_GAIN_LR * Cc.at<double>(0,0);
					
					// Front-Back
					tello_control[1] = TELLO_VS_GAIN_FB * Cc.at<double>(2,0);
					
					// Up-Down
					tello_control[2] = TELLO_VS_GAIN_UD * -Cc.at<double>(1,0);
					
					// Yaw
					tello_control[3] = TELLO_VS_GAIN_YAW * Cc.at<double>(3,0);
					
					// Display velocity screw
					snprintf( buf, 		TELLO_MAX_CHAR,
														"FPS=%llu Vx=% 3.2f Vy=% 3.2f Vz=% 3.2f wy=% 3.2f",
														frame_fps,
														Cc.at<double>(0,0),
														Cc.at<double>(1,0),
														Cc.at<double>(2,0),
														Cc.at<double>(3,0) );
					putText( l_frame, buf, 
													textOrg, 
													cv::FONT_HERSHEY_SIMPLEX, 
													TELLO_TXT_SIZE, 
													cv::Scalar::all(255), 
													TELLO_TXT_THICKNESS, 
													cv::LINE_AA );			
				}
				else
				{
					tello_servo = false;
					params.minArea = TELLO_MIN_BLOB_SIZE_HI;
					memset( tello_control, 0, sizeof( tello_control ) );
					putText( l_frame, "Some features are missng", 
													textOrg, 
													cv::FONT_HERSHEY_SIMPLEX, 
													TELLO_TXT_SIZE, 
													cv::Scalar::all(255), 
													TELLO_TXT_THICKNESS, 
													cv::LINE_AA );
				}
				
				// Display current local frame
				cv::imshow( TELLO_WINDOW_NAME, l_frame );
        
        // Get key pressed
        Key_pressed = cv::waitKey( 1 );
        
        #ifdef TELLO_FLIGHT
        // [QUESTION3] Manual control
        // 	-	UP/DOWN arrow keys: 		Tello FRONT-BACK
        //	-	RIGHT/LEFT arrows keys: Tello RIGHT-LEFT
        //	- q/w keys:								Tello UP-DOWN
        //	- =/: keys:								Tello YAW
        
        switch( Key_pressed )	{
					case TELLO_KEY_RIGHT:
						tello_control[0] = TELLO_MANUAL_V;
						break;
					case TELLO_KEY_LEFT:
						tello_control[0] = -TELLO_MANUAL_V;
						break;
					case TELLO_KEY_UP:
						tello_control[1] = TELLO_MANUAL_V;
						break;
					case TELLO_KEY_DOWN:
						tello_control[1] = -TELLO_MANUAL_V;
						break;
					case 'q':
						tello_control[2] = TELLO_MANUAL_V;
						break;
					case 'w':
						tello_control[2] = -TELLO_MANUAL_V;
						break;
					case '=':
						tello_control[3] = TELLO_MANUAL_V;
						break;
					case ':':
						tello_control[3] = -TELLO_MANUAL_V;
						break;
					default:
						break;
				}
        
				// Control signal saturation
        for ( int i = 0; i < TELLO_NB_DOF; i++ )	{
					if ( tello_control[i] > TELLO_VS_SAT )
						tello_control[i] = TELLO_VS_SAT;
					if ( tello_control[i] < -TELLO_VS_SAT )
						tello_control[i] = -TELLO_VS_SAT;
				}
				
				// Send control signal to Tello
				// LR+  : right
        // FB+  : front
        // UD+  : up
        // Yaw+ : clockwise when looking from the top
				tello.set_command_timeout( 1 );
				tello.move( tello_control[0], tello_control[1], tello_control[2], tello_control[3] );
				tello.set_command_timeout( TELLO_DEFAULT_COMMAND_TIMEOUT );
				#endif
        
        // If ESC is pressed, end loop
        if ( Key_pressed == TELLO_KEY_ESC ) {
          break;
        }
        
        // Wait for the next period
        usleep( TELLO_LOOP_PERIOD_US );
    }
    
    // Requesting thread termination
    tello_exit = true;
    pthread_join( frame_thread, NULL );
    
    printf( "[Tello] Closing stream window..." );
    cv::destroyWindow(TELLO_WINDOW_NAME);
    printf( "done\n" );
    
    #ifdef TELLO_FLIGHT
    printf( "[Tello] Drone landing..." );
    tello.land();
    printf( "done\n" );
    #endif
    
    printf( "[Tello] Closing video stream..." );
    tello.disable_video_stream();
    printf( "done\n" );
}
