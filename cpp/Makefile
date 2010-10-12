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
RELEASE_FLAGS = -arch i386
RELEASE_LINKFLAGS =  -arch i386 -dynamiclib 
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
