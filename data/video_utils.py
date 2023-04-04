import cv2

def count_frames(path):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0
	try:
		# check if we are using OpenCV 3
		total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	except:
		# otherwise, we are using OpenCV 2.4
		total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	# release the video file pointer
	video.release()
	# return the total number of frames in the video
	return total


def generate_video_frames(path, stride, start=0, end=None, convert_gray=True, resize=False, width=224, height=224, verbose=False):
	"""
	Receives a path to a video and a stride returns a generator yielding frames at a sampling rate of stride.

	Arguments:
		path {String} -- Path to the video file.
		stride {int} -- Stride of the video.

	Keyword Arguments:
		start {int} -- Start frame of the video. (default: {1})
		end {int} -- Last frame that should be used of the video. (default: {None})
		convert_gray {bool} -- Whether to convert the frames to gray scale. (default: {True})
		resize {bool} -- Resize the video to a given width and height. (default: {False})
		width {int} -- Width of the video. Applied if resize is True. (default: {224})
		height {int} -- Height of the video. Applied if resize is True. (default: {224})
		verbose {bool} -- Whether to print total number of frames available. (default: {False})

	Returns:
		video_list {list} -- List of frames loaded from the video.
	"""

	cap = cv2.VideoCapture(path)
	total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	if end is None:
		end = total_frames
	if verbose:
		print(f"Total frames: {total_frames} at stride: {stride} yield {(end-start)//stride+1} frames")
	frame_count = start
	# Set frame position to the start frame
	cap.set(cv2.CAP_PROP_POS_FRAMES, start)

	while cap.isOpened():

		ret, frame = cap.read()
		if not ret:
			break
		if resize:
			frame = cv2.resize(frame, (width, height))
		if convert_gray:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# cv2.imwrite('frame{:d}.jpg'.format(count), frame)
		yield frame

		frame_count += stride
		# If the frame count is greater than the end frame, break
		if frame_count > end:
			break
		# Else set the frame position to the next frame
		else:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

	cap.release()

def load_video(path: str, stride: int, start: int=0, end: int=None, convert_gray: bool=True, resize: bool=False, width: int=224, height: int=224, verbose: bool=False):
	"""
	Receives a path to a video and a stride returns a list of frames at a sampling rate of stride.
	Wrapper function for generate_video_frames.

	Arguments:
		path {String} -- Path to the video file.
		stride {int} -- Stride of the video.

	Keyword Arguments:
		start {int} -- Start frame of the video. (default: {1})
		end {int} -- Last frame that should be used of the video. (default: {None})
		convert_gray {bool} -- Whether to convert the frames to gray scale. (default: {True})
		resize {bool} -- Resize the video to a given width and height. (default: {False})
		width {int} -- Width of the video. Applied if resize is True. (default: {224})
		height {int} -- Height of the video. Applied if resize is True. (default: {224})
		verbose {bool} -- Whether to print total number of frames available and frames loaded. (default: {False})

	Returns:
		video_list {list} -- List of frames loaded from the video.
	"""

	video_generator = generate_video_frames(path, stride, start, end, convert_gray, resize, width, height, verbose)
	video_list = [x for x in video_generator]
	if verbose:
		print("Frames loaded: {}".format(len(video_list)))
	return video_list