set_global_assignment -entity "mult_add_fix8bx2" -library "lib_mult_add_fix8bx2" -name IP_TOOL_NAME "altera_mult_add"
set_global_assignment -entity "mult_add_fix8bx2" -library "lib_mult_add_fix8bx2" -name IP_TOOL_VERSION "16.0"
set_global_assignment -entity "mult_add_fix8bx2" -library "lib_mult_add_fix8bx2" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_mult_add_fix8bx2" -name SPD_FILE [file join $::quartus(sip_path) "mult_add_fix8bx2.spd"]

set_global_assignment -library "lib_mult_add_fix8bx2" -name MISC_FILE [file join $::quartus(sip_path) "mult_add_fix8bx2_sim/mult_add_fix8bx2.v"]
