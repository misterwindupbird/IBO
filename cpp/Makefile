# Copyright (C) 2011 by Eric Brochu
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

UNAME = $(shell uname)

INCLUDE_PATH = -I.
DEPEND = g++
RELEASE_FLAGS = -O3
DEBUG_FLAGS = -g -O0
CC = g++
LINK = g++

ifeq ($(UNAME), Linux)
RELEASE_FLAGS = -Wall
RELEASE_LINKFLAGS =  -shared -W1,-soname,libego.so.1 
endif
ifeq ($(UNAME), Darwin)
RELEASE_FLAGS = 
RELEASE_LINKFLAGS =  -dynamiclib 
endif

# Source

SRC = direct.cpp optimizeRF.cpp optimizeGP.cpp helpers.cpp

# object files
RELEASE_OBJ = $(patsubst %.cpp,obj/%.o,$(notdir $(SRC)))
DEBUG_OBJ = $(patsubst %.cpp,obj_debug/%.o,$(notdir $(SRC)))

# how to make the main target (debug mode, the default)
DYLIB = libs/libego.dylib
$(DYLIB): $(RELEASE_OBJ)
	$(LINK) $(RELEASE_LINKFLAGS) -o $@ $^ $(LINK_LIBS)
# $(DYLIB): $(DEBUG_OBJ)
#   $(LINK) $(DEBUG_LINKFLAGS) -dynamiclib -o $@ $^ $(LINK_LIBS)

# how to compile each file
.SUFFIXES:
obj/%.o:
	$(CC) -c $(RELEASE_FLAGS) $(INCLUDE_PATH) -o $@ $<
obj_debug/%.o:
	$(CC) -c $(DEBUG_FLAGS) $(INCLUDE_PATH) -o $@ $<

# cleaning up
.PHONY: clean
clean:
	-rm -f obj/*.o obj/depend $(DYLIB) obj_debug/*.o obj_debug/depend $(MAIN_PROGRAM)_release *core 
	
# dependencies are automatically generated
.PHONY: depend
depend:
	-mkdir libs
	-mkdir obj
	-rm -f obj/depend
	$(foreach srcfile,$(SRC),$(DEPEND) -MM $(srcfile) -MT $(patsubst %.cpp,obj/%.o,$(notdir $(srcfile))) $(INCLUDE_PATH) >> obj/depend;)
	-mkdir obj_debug
	-rm -f obj_debug/depend
	$(foreach srcfile,$(SRC),$(DEPEND) -MM $(srcfile) -MT $(patsubst %.cpp,obj_debug/%.o,$(notdir $(srcfile))) $(INCLUDE_PATH) >> obj_debug/depend;)

-include obj/depend
-include obj_debug/depend
