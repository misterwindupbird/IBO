DIRS	= cpp
ECHO	= echo

all : force_look
	$(ECHO) looking into subdir .
	-for d in $(DIRS); do (cd $$d; $(MAKE) ); done

depend :
	$(ECHO) make dependencies in .
	-for d in $(DIRS); do (cd $$d; $(MAKE) depend ); done

clean :
	$(ECHO) cleaning up in .
	-for d in $(DIRS); do (cd $$d; $(MAKE) clean ); done

force_look :
	true