# Subsystem Decomposition

![Autonomous-Agent-Subsystems](https://user-images.githubusercontent.com/42655977/212421992-068d2c7c-97f1-4f4c-9219-86e082bdd669.jpg)


# Execution

**Notice**: The following steps assume familiarity with [Duckietown Project](https://www.duckietown.org). Please refer to the [Duckiebot Manual](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/index.html) for more details on Hardware/Software requirements to run this agent.

1.Build docker image on duckiebot

Run command

`dts devel build -f -H <robot_name>.local`

Replace `<robot_name>` with name of your duckiebot.

2.Start docker container

Run command

`dts devel run -H <robot_name>.local`

Replace `<robot_name>` with name of your duckiebot.

