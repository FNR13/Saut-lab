% Load the bag
bag = rosbag('lab2testWithCamera.bag');

% Select the /pose topic
bagPose = select(bag, 'Topic', '/pose');

% Read messages
msgStructsPose = readMessages(bagPose, 'DataFormat', 'struct');

% Timestamps
timestampsPose = bagPose.MessageList.Time;

% Process /pose (Odometry) - only X, Y and orientation
n_pose = numel(msgStructsPose);
pose_vectors = zeros(n_pose, 6);  % X, Y, qX, qY, qZ, qW
for i = 1:n_pose
    msg = msgStructsPose{i};
    pose_vectors(i, :) = [
        msg.Pose.Pose.Position.X, msg.Pose.Pose.Position.Y, ...
        msg.Pose.Pose.Orientation.X, msg.Pose.Pose.Orientation.Y, ...
        msg.Pose.Pose.Orientation.Z, msg.Pose.Pose.Orientation.W];
end

% Shift initial position to (0,0)
initialX = pose_vectors(1,1);
initialY = pose_vectors(1,2);
pose_vectors(:,1) = pose_vectors(:,1) - initialX;
pose_vectors(:,2) = pose_vectors(:,2) - initialY;

% Plot odometry path (2D)
figure;
plot(pose_vectors(:,1), pose_vectors(:,2), 'LineWidth', 1.5);
xlabel('X'); ylabel('Y');
title('2D Odometry Path (/pose) - Origin Shifted');
grid on; axis equal;

% Save pose data
save("pose_only.mat", "timestampsPose", "pose_vectors");

