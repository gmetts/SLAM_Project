
file(GLOB_RECURSE CALIB_SOURCES "src/*.cpp")

add_executable(camera_calib ${CALIB_SOURCES})

target_link_libraries(camera_calib
	${OpenCV_LIBS}
	${ZBAR_LIBRARIES}
	QRPoseEstimator
	)
