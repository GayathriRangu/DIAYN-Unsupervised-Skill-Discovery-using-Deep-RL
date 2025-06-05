all:
	cl /O2 /MT /EHsc /I../include simulate.cpp ../bin/mujoco131.lib ../bin/glfw3.lib /Fe../bin/
	cl /O2 /MT /EHsc /I../include test.cpp ../bin/mujoco131.lib /Fe../bin/
	cl /O2 /MT /EHsc /I../include compile.cpp ../bin/mujoco131.lib /Fe../bin/
	del *.obj
