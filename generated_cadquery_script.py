You can also use the 'cq.Workplane('XY')' to create the workplane.
```python
import cadquery as cq

w0 = cq.Workplane('XY')

# Create a 5x5x5 cube
cube = w0.box(5, 5, 5)

# Add a .25 fillet to all edges
cube = cube.fillet(0.25)

# Add a hole to the center of the cube's top face
hole_center = w0.size()[0] / 2
hole = w0.circle(2).translate((hole_center, hole_center, 0))
cube = cube.cut(hole)

# Add another hole to the left face
hole_left = w0.size()[0] / 2
hole = w0.circle(2).translate((-hole_left, 0, 0))
cube = cube.cut(hole)

# Show the resulting object
show_object(cube)
```
This script creates a 5x5x5 cube and adds a .25 fillet to all its edges. It then adds a hole to the center of the cube's top face and another hole to the left face. Both hole diameters are 2. The resulting object is then displayed using the `show_object` function.