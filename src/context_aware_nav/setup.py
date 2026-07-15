from setuptools import find_packages, setup

package_name = 'context_aware_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tapaswee',
    maintainer_email='dasaritapaswee2018@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'context_planner = context_aware_nav.context_aware_planner:main',
            'arm_commander = context_aware_nav.arm_commander:main',
            'start = context_aware_nav.start:main',
            'arm_test_node = context_aware_nav.arm_test_node:main',
            'smart_nav = context_aware_nav.smart_nav:main',
            'smart_nav_node = context_aware_nav.smart_nav_node:main',
            'yolo_context = context_aware_nav.yolo_context:main',
            'smart_nav_vision = context_aware_nav.smart_nav_vision:main',
            'smart_nav_vision2 = context_aware_nav.smart_nav_vision2:main',
            'grasp_node= context_aware_nav.grasp_node:main',
            'wp_markers= context_aware_nav.waypoint_markers:main',
            'context_mgr= context_aware_nav.context_manager:main',
        ],
    },
)
