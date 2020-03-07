`timescale 1 ps / 1 ps
module mult_fix8b (
		input   clock,
		input   resetn,
		input   ivalid, 
		input   iready,
		output  ovalid, 
		output  oready,
		input  wire [7:0]  dataa_0, // dataa_0.dataa_0
		input  wire [7:0]  datab_0, // datab_0.datab_0
		output wire [15:0] result  //  result.result
	);

	assign ovalid = 1'b1;
	assign oready = 1'b1;
	// ivalid, iready, resetn are ignored

	mult_fix8b_0002 mult_fix8b_inst (
		.result  (result),  //  result.result
		.dataa_0 (dataa_0), // dataa_0.dataa_0
		.datab_0 (datab_0), // datab_0.datab_0
		.clock0  (clock)   //  clock0.clock0
	);

endmodule
