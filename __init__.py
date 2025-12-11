"""
GRACE Data Processor Plugin for QGIS
Author: M. G. Gurmu
Email: mggurmu@gmail.com
Version: 1.0.0
"""

import os
from qgis.PyQt.QtGui import QIcon

def classFactory(iface):
    from .grace_processor_qgis import GraceProcessorPlugin
    return GraceProcessorPlugin(iface)

# Plugin metadata
__version__ = "1.0.0"
__author__ = "M. G. Gurmu"
__email__ = "mggurmu@gmail.com"

# Get plugin directory
plugin_dir = os.path.dirname(__file__)

# Define icon path
def icon():
    """Return the plugin icon"""
    icon_path = os.path.join(plugin_dir, 'icon.png')
    if os.path.exists(icon_path):
        return QIcon(icon_path)
    else:
        # Return a default icon if file doesn't exist
        return QIcon()