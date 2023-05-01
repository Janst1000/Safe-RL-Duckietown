#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
#dt-exec roslaunch lane_filter lane_filter_node.launch veh:=$VEHICLE_NAME
# launch lane_filter_node in background
#roslaunch lane_filter lane_filter_node.launch veh:=$VEHICLE_NAME &
dt-exec roslaunch rl_agent test_env.launch veh:=$VEHICLE_NAME
#roslaunch rl_agent collect_data.launch veh:=$VEHICLE_NAME
#bash

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
