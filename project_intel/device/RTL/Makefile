# Remember to recompile the lib everytime you changed the rtl

lib:
	aocl library hdl-comp-pkg rtl_lib.xml -o rtl_lib.aoco
	aocl library create -name rtl_lib rtl_lib.aoco

clean:
	rm -rf *.aoco *.aoclib
