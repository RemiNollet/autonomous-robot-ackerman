from setuptools import setup

package_name = 'carsim_bridge'

# Standard ament_python layout: carsim_bridge/carsim_bridge/ holds the
# actual module, this file (carsim_bridge/setup.py) is the colcon package
# root. An earlier version of this file used package_dir={package_name:
# '.'} to keep the .py files flat in this directory -- switched to the
# standard nested layout after colcon build --symlink-install failed on
# the flat mapping (colcon_core.task.python.build assumes the nested
# convention when building symlinks). tests/test_perception_node.py and
# tests/test_perception_inference.py were updated to match (second
# sys.path entry pointing at this directory, alongside the existing one
# for perception/).
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Remi Nollet',
    maintainer_email='remi.nollet@live.fr',
    description=(
        'ZeroMQ bridge to the macOS MuJoCo sim, dummy controller, and the '
        'lane-perception node.'
    ),
    license='MIT',
    entry_points={
        'console_scripts': [
            'bridge_node = carsim_bridge.bridge_node:main',
            'dummy_controller_node = carsim_bridge.dummy_controller_node:main',
            'perception_node = carsim_bridge.perception_node:main',
        ],
    },
)
