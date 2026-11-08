# Copyright 2022 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


def generate_launch_description():
    # Configure ROS nodes for launch

    gz_ln_arg = DeclareLaunchArgument(
        'gazebo_launch',
        default_value='True'
    )
    camera_view_arg = DeclareLaunchArgument(
        'camera_view',
        default_value='True',
        description='Open an rqt_image_view window for the AI-deck camera.'
    )
    impairment_gui_arg = DeclareLaunchArgument(
        'impairment_gui',
        default_value='True',
        description='Open the separate delay/jitter/loss configuration window.'
    )
    perception_arg = DeclareLaunchArgument(
        'baseline_perception',
        default_value='True',
        description='Run passive single-frame tutor-wire perception.'
    )
    perception_log_arg = DeclareLaunchArgument(
        'perception_log_path',
        default_value='/tmp/crazyflie_perception_baseline.csv',
        description='CSV path for passive perception observations.'
    )
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='greenhouse_world.sdf',
        description='Gazebo world filename installed by ros_gz_crazyflie_gazebo.'
    )
    # Setup project paths
    pkg_project_bringup = get_package_share_directory('ros_gz_crazyflie_bringup')
    pkg_project_gazebo = get_package_share_directory('ros_gz_crazyflie_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Load the SDF file from "description" package
    if LaunchConfiguration('gazebo_launch') == 'True':
        gz_model_path = os.getenv('GZ_SIM_RESOURCE_PATH')
        sdf_file  =  os.path.join(gz_model_path, 'crazyflie', 'model.sdf')
        with open(sdf_file, 'r') as infp:
            robot_desc = infp.read()

    # Setup to launch the simulator and Gazebo world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            condition=IfCondition(LaunchConfiguration('gazebo_launch')),
        launch_arguments={
            'gz_version': '6',
            'gz_args': [
                PathJoinSubstitution([
                    pkg_project_gazebo,
                    'worlds',
                    LaunchConfiguration('world')
                ]),
                ' -r'
            ]
        }.items(),
    )


    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(pkg_project_bringup, 'config', 'ros_gz_crazyflie_bridge.yaml'),
        }],

        output='screen'
    )

    control = Node(
        package='ros_gz_crazyflie_control',
        executable='control_services',
        output='screen',
        respawn=True,
        respawn_delay=1.0,
        parameters=[
            {'hover_height': 0.5},
            {'robot_prefix': '/crazyflie'},
            {'incoming_twist_topic': '/cmd_vel'},
            {'max_ang_z_rate': 0.4},
            {'max_vertical_speed': 0.3},
            {'altitude_kp': 2.0},
            {'altitude_ki': 0.35},
            {'altitude_kd': 0.35},
        ]
    )

    flow_deck = Node(
        package='ros_gz_crazyflie_control',
        executable='flow_deck_simulator',
        output='screen',
        respawn=True,
        respawn_delay=1.0,
        parameters=[
            {'robot_prefix': '/crazyflie'},
            {'focal_length_px': 160.0},
        ]
    )

    camera_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        arguments=['/crazyflie/camera/image'],
        condition=IfCondition(LaunchConfiguration('camera_view')),
        output='screen',
    )

    impairment = Node(
        package='ros_gz_crazyflie_control',
        executable='image_impairment',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    impairment_gui = Node(
        package='ros_gz_crazyflie_control',
        executable='image_impairment_gui',
        condition=IfCondition(LaunchConfiguration('impairment_gui')),
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    perception = Node(
        package='ros_gz_crazyflie_control',
        executable='baseline_perception',
        condition=IfCondition(LaunchConfiguration('baseline_perception')),
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/crazyflie/camera/image_delayed',
            'log_path': LaunchConfiguration('perception_log_path'),
        }],
    )

    return LaunchDescription([
        gz_ln_arg,
        camera_view_arg,
        impairment_gui_arg,
        perception_arg,
        perception_log_arg,
        world_arg,
        gz_sim,
        bridge,
        control,
        flow_deck,
        camera_view,
        impairment,
        impairment_gui,
        perception,
    ])
