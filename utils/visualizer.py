"""
ROS-free visualizer module for PyMPC.

This module provides visualization capabilities without ROS dependencies.
It includes mock implementations of ROS visualization markers for use
in standalone Python environments.
"""

from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class Colormap(Enum):
    VIRIDIS = 1
    INFERNO = 2
    BRUNO = 3


class MockVisualizer:
    """Mock visualizer that provides ROS-like interface without ROS dependencies."""
    
    def __init__(self, name: str = "mock_visualizer", frame_id: str = "map"):
        self.name = name
        self.frame_id = frame_id
        self.data = []
        
    def publish(self, data=None):
        """Mock publish method - stores data for later visualization."""
        if data is not None:
            if not hasattr(self, 'data') or self.data is None:
                self.data = []
            self.data.append(data)
    
    def clear(self):
        """Clear stored data."""
        self.data = []


class ROSLine(MockVisualizer):
    """ROS-free line visualizer."""
    
    def __init__(self, name: str = "line", frame_id: str = "map"):
        super().__init__(name, frame_id)
        self.points = []
        
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the line."""
        self.points.append([x, y, z])
    
    def set_color(self, r: float, g: float, b: float, alpha: float = 1.0):
        """Set line color."""
        self.color = (r, g, b, alpha)
    
    def set_width(self, width: float):
        """Set line width."""
        self.width = width
    
    def publish(self):
        """Publish the line (store for visualization)."""
        if len(self.points) > 1:
            self.data = {
                'type': 'line',
                'points': self.points,
                'color': getattr(self, 'color', (1.0, 0.0, 0.0, 1.0)),
                'width': getattr(self, 'width', 1.0)
            }


class ROSMarker(MockVisualizer):
    """ROS-free marker visualizer."""
    
    # Color palettes
    VIRIDIS = [253, 231, 37, 234, 229, 26, 210, 226, 27, 186, 222, 40, 162, 218, 55, 139, 214, 70, 119, 209, 83, 99,
               203, 95, 80, 196, 106, 63, 188, 115, 49, 181, 123, 38, 173, 129, 33, 165, 133, 30, 157, 137, 31, 148,
               140, 34, 140, 141, 37, 131, 142, 41, 123, 142, 44, 115, 142, 47, 107, 142, 51, 98, 141, 56, 89, 140]
    INFERNO = [252, 255, 164, 241, 237, 113, 246, 213, 67, 251, 186, 31, 252, 161, 8, 248, 135, 14, 241, 113, 31, 229,
               92, 48, 215, 75, 63, 196, 60, 78, 177, 50, 90, 155, 41, 100, 135, 33, 107, 113, 25, 110, 92, 18, 110, 69,
               10, 105, 47, 10, 91, 24, 12, 60]
    BRUNO = [217, 83, 25, 0, 114, 189, 119, 172, 48, 126, 47, 142, 237, 177, 32, 77, 190, 238, 162, 19, 47, 256, 153,
             256, 0, 103, 256]

    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__("marker", frame_id)
        self.publisher = publisher
        self.marker_data = {}

    def stamp(self):
        """Mock timestamp."""
        import time
        self.marker_data['timestamp'] = time.time()

    def set_color(self, r: float, g: float, b: float, alpha: float = 1.0):
        """Set marker color."""
        self.marker_data['color'] = (r, g, b, alpha)

    def set_color_ratio(self, ratio: float, alpha: float = 1.0):
        """Set color based on ratio."""
        r, g, b = self.get_color_from_range(ratio)
        self.set_color(r, g, b, alpha)

    def set_color_int(self, select: int, alpha: float = 1.0, colormap: Colormap = Colormap.VIRIDIS):
        """Set color based on integer selection."""
        r, g, b = self.get_color_from_range_int(select, colormap)
        self.set_color(r, g, b, alpha)

    def get_color_from_range(self, ratio: float) -> Tuple[float, float, float]:
        """Get color from ratio (0.0 to 1.0)."""
        # Convert ratio to RGB using viridis colormap
        colors = self.VIRIDIS
        idx = int(ratio * (len(colors) // 3 - 1)) * 3
        idx = max(0, min(idx, len(colors) - 3))
        
        r = colors[idx] / 255.0
        g = colors[idx + 1] / 255.0
        b = colors[idx + 2] / 255.0
        
        return r, g, b

    def get_color_from_range_int(self, select: int, colormap: Colormap = Colormap.VIRIDIS) -> Tuple[float, float, float]:
        """Get color from integer selection."""
        if colormap == Colormap.VIRIDIS:
            colors = self.VIRIDIS
        elif colormap == Colormap.INFERNO:
            colors = self.INFERNO
        else:
            colors = self.BRUNO
            
        idx = (select % (len(colors) // 3)) * 3
        r = colors[idx] / 255.0
        g = colors[idx + 1] / 255.0
        b = colors[idx + 2] / 255.0
        
        return r, g, b

    def set_scale(self, x: float, y: float, z: float):
        """Set marker scale."""
        self.marker_data['scale'] = (x, y, z)

    def set_position(self, x: float, y: float, z: float = 0.0):
        """Set marker position."""
        self.marker_data['position'] = (x, y, z)

    def set_orientation(self, x: float, y: float, z: float, w: float):
        """Set marker orientation (quaternion)."""
        self.marker_data['orientation'] = (x, y, z, w)

    def set_text(self, text: str):
        """Set marker text."""
        self.marker_data['text'] = text

    def set_type(self, marker_type: int):
        """Set marker type."""
        self.marker_data['type'] = marker_type

    def publish(self):
        """Publish marker (store for visualization)."""
        self.data = self.marker_data.copy()


class ROSMarkerPublisher(MockVisualizer):
    """ROS-free marker publisher."""
    
    def __init__(self, name: str = "marker_publisher", frame_id: str = "map"):
        super().__init__(name, frame_id)
        self.markers = []
        
    def add_marker(self, marker: ROSMarker):
        """Add a marker to the publisher."""
        self.markers.append(marker)

    def publish(self):
        """Publish all markers."""
        marker_data = []
        for marker in self.markers:
            if hasattr(marker, 'data') and marker.data:
                marker_data.append(marker.data)
        self.data = marker_data


class ROSPointMarker(ROSMarker):
    """ROS-free point marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(2)  # SPHERE
        self.set_scale(0.1, 0.1, 0.1)


class ROSMultiplePointMarker(ROSMarker):
    """ROS-free multiple point marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(8)  # POINTS
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the marker."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish multiple points."""
        self.marker_data['points'] = self.points
        super().publish()


class ROSTextMarker(ROSMarker):
    """ROS-free text marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(9)  # TEXT_VIEW_FACING


class ROSPolygonMarker(ROSMarker):
    """ROS-free polygon marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(5)  # LINE_LIST
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the polygon."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish polygon."""
        self.marker_data['points'] = self.points
        super().publish()


class ROSArrowMarker(ROSMarker):
    """ROS-free arrow marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(0)  # ARROW


class ROSCubeMarker(ROSMarker):
    """ROS-free cube marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(1)  # CUBE


class ROSSphereMarker(ROSMarker):
    """ROS-free sphere marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(2)  # SPHERE


class ROSCylinderMarker(ROSMarker):
    """ROS-free cylinder marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(3)  # CYLINDER


class ROSPlaneMarker(ROSMarker):
    """ROS-free plane marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(4)  # PLANE


class ROSLineStripMarker(ROSMarker):
    """ROS-free line strip marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(4)  # LINE_STRIP
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the line strip."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish line strip."""
        self.marker_data['points'] = self.points
        super().publish()


class ROSLineListMarker(ROSMarker):
    """ROS-free line list marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(5)  # LINE_LIST
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the line list."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish line list."""
        self.marker_data['points'] = self.points
        super().publish()


class ROSCubeListMarker(ROSMarker):
    """ROS-free cube list marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(6)  # CUBE_LIST
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the cube list."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish cube list."""
        self.marker_data['points'] = self.points
        super().publish()


class ROSSphereListMarker(ROSMarker):
    """ROS-free sphere list marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(7)  # SPHERE_LIST
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the sphere list."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish sphere list."""
        self.marker_data['points'] = self.points
        super().publish()


class ROSPointsMarker(ROSMarker):
    """ROS-free points marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(8)  # POINTS
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the points marker."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish points."""
        self.marker_data['points'] = self.points
        super().publish()


class ROSMeshMarker(ROSMarker):
    """ROS-free mesh marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(10)  # MESH_RESOURCE


class ROSTriangleListMarker(ROSMarker):
    """ROS-free triangle list marker."""
    
    def __init__(self, publisher=None, frame_id: str = "map"):
        super().__init__(publisher, frame_id)
        self.set_type(11)  # TRIANGLE_LIST
        self.points = []
    
    def add_point(self, x: float, y: float, z: float = 0.0):
        """Add a point to the triangle list."""
        self.points.append((x, y, z))
    
    def publish(self):
        """Publish triangle list."""
        self.marker_data['points'] = self.points
        super().publish()


class VisualizationManager:
    """Manager for ROS-free visualization."""
    
    def __init__(self):
        self.visualizers = {}
        self.fig = None
        self.ax = None
        
    def add_visualizer(self, name: str, visualizer: MockVisualizer):
        """Add a visualizer."""
        self.visualizers[name] = visualizer
        
    def create_plot(self, title: str = "PyMPC Visualization", figsize: Tuple[int, int] = (12, 8)):
        """Create a matplotlib plot."""
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(title)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
    def visualize_all(self):
        """Visualize all stored data."""
        if self.fig is None:
            self.create_plot()
            
        for name, visualizer in self.visualizers.items():
            if hasattr(visualizer, 'data') and visualizer.data:
                self._plot_visualizer_data(name, visualizer.data)
                
        plt.legend()
        plt.show()
        
    def _plot_visualizer_data(self, name: str, data):
        """Plot data from a visualizer."""
        if isinstance(data, dict):
            if data.get('type') == 'line' and 'points' in data:
                points = np.array(data['points'])
                if len(points) > 1:
                    color = data.get('color', (1.0, 0.0, 0.0, 1.0))[:3]
                    width = data.get('width', 1.0)
                    self.ax.plot(points[:, 0], points[:, 1], color=color, linewidth=width, label=name)
                    
            elif 'position' in data:
                pos = data['position']
                color = data.get('color', (1.0, 0.0, 0.0, 1.0))[:3]
                scale = data.get('scale', (0.1, 0.1, 0.1))
                
                if data.get('type') == 2:  # SPHERE
                    circle = patches.Circle((pos[0], pos[1]), scale[0], color=color, alpha=0.7)
                    self.ax.add_patch(circle)
                elif data.get('type') == 1:  # CUBE
                    rect = patches.Rectangle((pos[0] - scale[0]/2, pos[1] - scale[1]/2), 
                                          scale[0], scale[1], color=color, alpha=0.7)
                    self.ax.add_patch(rect)
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._plot_visualizer_data(f"{name}_{i}", item)
                
    def save_plot(self, filename: str):
        """Save the current plot."""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            
    def clear_all(self):
        """Clear all visualizer data."""
        for visualizer in self.visualizers.values():
            visualizer.clear()


# Global visualization manager
viz_manager = VisualizationManager()


def create_visualization_publisher(name: str, publisher_type=ROSLine, frame_id: str = "map"):
    """Create a visualization publisher (ROS-free version)."""
    publisher = publisher_type(name, frame_id)
    viz_manager.add_visualizer(name, publisher)
    return publisher