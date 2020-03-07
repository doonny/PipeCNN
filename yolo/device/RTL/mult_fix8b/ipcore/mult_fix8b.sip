set_global_assignment -entity "mult_fix8b" -library "lib_mult_fix8b" -name IP_TOOL_NAME "altera_mult_add"
set_global_assignment -entity "mult_fix8b" -library "lib_mult_fix8b" -name IP_TOOL_VERSION "16.0"
set_global_assignment -entity "mult_fix8b" -library "lib_mult_fix8b" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_mult_fix8b" -name SPD_FILE [file join $::quartus(sip_path) "mult_fix8b.spd"]

set_global_assignment -library "lib_mult_fix8b" -name MISC_FILE [file join $::quartus(sip_path) "mult_fix8b_sim/mult_fix8b.v"]
