from setuptools import setup

package_name = 'carsim_bridge'

# package_dir maps the package straight onto this directory instead of the
# usual nested carsim_bridge/carsim_bridge/ ament_python layout. Deliberate:
# every existing import (this package's own node files, and every test in
# ../tests/) already assumes autonomous-robot-ackerman/carsim_bridge/ *is*
# the importable carsim_bridge package -- see tests/test_perception_node.py
# and tests/test_perception_inference.py, both written and passing against
# this flat layout before this file existed. Nesting would move every .py
# file and update every one of those imports for no functional gain.
setup(
    name=package_name,
    version='0.1.0',
    package_dir={package_name: '.'},
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Remi Nollet',
    maintainer_email='you@example.com',
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
