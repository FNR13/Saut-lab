% Load the bag
bag = rosbag('lab2testWithId.bag');

% Select the camera topic
bagImg = select(bag, 'Topic', '/fiducial_images');

% Read image messages
msgs = readMessages(bagImg);

% Get number of frames
n = length(msgs);
disp(['Found ', num2str(n), ' image frames.']);

% Create a VideoWriter object
outputVideo = VideoWriter('camera_video.avi');
outputVideo.FrameRate = 30;  % Adjust frame rate as needed
open(outputVideo);

% Write frames to video
for i = 1:n
    img = readImage(msgs{i});
    writeVideo(outputVideo, img);
end

% Finalize video file
close(outputVideo);

disp('Video saved as camera_video.avi');
