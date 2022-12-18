#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
# roscore already running on the bot no need to keep it when deploying to the bot
roscore &
sleep 5
# dt-exec rosrun utilities state_machine.py
# dt-exec rosrun utilities state_changer_mock.py
# dt-exec rosrun utilities state_subscriber_mock.py
# dt-exec rosrun utilities http_duckie_server.py
# dt-exec python3 ./packages/utilities/src/http_tester.py

# launching app
dt-exec roslaunch utilities all.launch \
    veh:="$VEHICLE_NAME" \
    robot_type:="$ROBOT_TYPE" \
    robot_configuration:="$ROBOT_CONFIGURATION"

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
