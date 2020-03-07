`timescale 1 ps / 1 ps
module mult_add_fix8bx2 (
		input   clock,
		input   resetn,
		input   ivalid, 
		input   iready,
		output  ovalid, 
		output  oready,
		
		input [7:0]  dataa_0, // dataa_0.dataa_0
		input [7:0]  datab_0, // datab_0.datab_0
		
		input [7:0]  dataa_1, // dataa_1.dataa_1
		input [7:0]  datab_1, // datab_1.datab_1
		
		output[31:0] result   //  result.result
		);


	wire [16:0] result_17b;

	assign ovalid = 1'b1;
	assign oready = 1'b1;
	// ivalid, iready, resetn are ignored

	assign result = {{15{result_17b[16]}}, result_17b};
	//assign result = {15'd0, result_17b};

	mult_add_fix8bx2_0002 mult_add_fix8bx2_inst (
		.result  (result_17b),  //  result.result
		.dataa_0 (dataa_0), // dataa_0.dataa_0
		.dataa_1 (dataa_1), // dataa_1.dataa_1
		.datab_0 (datab_0), // datab_0.datab_0
		.datab_1 (datab_1), // datab_1.datab_1
		.clock0  (clock)   //  clock0.clock0
	);

endmodule