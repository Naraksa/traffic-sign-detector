import sys
import inspect

print("Python executable:", sys.executable)
print("inspect module path:", inspect.__file__)
print("Has signature:", hasattr(inspect, "signature"))