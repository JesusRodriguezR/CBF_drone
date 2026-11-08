from setuptools import find_packages, setup

package_name = 'ros_gz_crazyflie_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/monitoring_points.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kimberly McGuire',
    maintainer_email='kimberly@bitcraze.io',
    description='Simple control scripts for the Crazyflie',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_services = ros_gz_crazyflie_control.control_services:main',
            'flow_deck_simulator = ros_gz_crazyflie_control.flow_deck_simulator:main',
            'greenhouse_fsm_trial = ros_gz_crazyflie_control.fsm_trial:main',
            'greenhouse_fsm_visualizer = ros_gz_crazyflie_control.fsm_visualizer:main',
            'image_impairment = ros_gz_crazyflie_control.image_impairment:main',
            'image_impairment_gui = ros_gz_crazyflie_control.impairment_gui:main',
            'baseline_perception = ros_gz_crazyflie_control.perception_baseline:main',
        ],
    },
)
