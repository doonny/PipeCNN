set_global_assignment -entity "mult_add_fix8bx4" -library "lib_mult_add_fix8bx4" -name IP_TOOL_NAME "altera_mult_add"
set_global_assignment -entity "mult_add_fix8bx4" -library "lib_mult_add_fix8bx4" -name IP_TOOL_VERSION "16.0"
set_global_assignment -entity "mult_add_fix8bx4" -library "lib_mult_add_fix8bx4" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_mult_add_fix8bx4" -name SPD_FILE [file join $::quartus(sip_path) "mult_add_fix8bx4.spd"]

set_global_assignment -library "lib_mult_add_fix8bx4" -name MISC_FILE [file join $::quartus(sip_path) "mult_add_fix8bx4_sim/mult_add_fix8bx4.v"]
