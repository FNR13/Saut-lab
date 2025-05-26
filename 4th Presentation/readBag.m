clear
clc
close all
% Load the bag
bag = rosbag('lab2testWithId.bag');


% Select the /pose topic
bagPose = select(bag, 'Topic', '/pose');
% Select the Identification topic
bagObs = select(bag, 'Topic', '/fiducial_transforms');

% Read messages
msgStructsPose = readMessages(bagPose, 'DataFormat', 'struct');
msgStructsObs = readMessages(bagObs, 'DataFormat', 'struct');

% Timestamps
timestampsPose = bagPose.MessageList.Time;
timestampsObs = bagObs.MessageList.Time;

% Process /pose (Odometry) - only X, Y and orientation, linear and angular velocity 
n_pose = numel(msgStructsPose);
pose_vectors = zeros(n_pose, 5);
for i = 1:n_pose
    msg = msgStructsPose{i};
    pose_vectors(i, :) = [
        msg.Pose.Pose.Position.X, msg.Pose.Pose.Position.Y, msg.Pose.Pose.Orientation.Z, ...
        msg.Twist.Twist.Linear.X, msg.Twist.Twist.Angular.Z ];

end

% Preallocate a cell array or grow dynamically
obs_data = [];  % will hold rows: [timestamp, FiducialId, X, Y, Z]

for i = 1:n_pose
    msg = msgStructsObs{i};
    t = timestampsPose(i);  % timestamp corresponding to this message
    
    n_markers = numel(msg.Transforms);
    for j = 1:n_markers
        obs_data = [obs_data; ...
            datenum(t), ...  % Convert datetime to serial number (for numeric storage)
            msg.Transforms(j).FiducialId, ...
            msg.Transforms(j).Transform.Translation.X, ...
            msg.Transforms(j).Transform.Translation.Y, ...
            msg.Transforms(j).Transform.Translation.Z];
    end
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

% Extract time in seconds relative to the start
time_seconds = seconds(timestampsPose - timestampsPose(1));

% Extract speeds
linear_speed = pose_vectors(:, 4);   % Linear X speed
angular_speed = pose_vectors(:, 5);  % Angular Z speed

% Plot both speeds
figure;
yyaxis left
plot(time_seconds, linear_speed, 'b', 'LineWidth', 1.5);
ylabel('Linear Speed (m/s)');
yyaxis right
plot(time_seconds, angular_speed, 'r', 'LineWidth', 1.5);
ylabel('Angular Speed (rad/s)');
xlabel('Time (s)');
title('Linear and Angular Speeds over Time');
legend('Linear Speed', 'Angular Speed');
grid on;


% Save pose data
save("pose_only.mat", "timestampsPose", "pose_vectors");

