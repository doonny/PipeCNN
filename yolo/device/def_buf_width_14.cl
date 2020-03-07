#ifndef _DEF_BUF_WIDTH
#define _DEF_BUF_WIDTH

#define DEFINE_BUF(buffer_name, buffer_depth)\
	buffer_name##_0[buffer_depth][8],\
	buffer_name##_1[buffer_depth][4],\
	buffer_name##_2[buffer_depth][2]\

#define WRITE_BUF(buffer_name, addr_name, data_name)\
{\
	buffer_name##_0[addr_name][0] = data_name.lane[0];\
	buffer_name##_0[addr_name][1] = data_name.lane[1];\
	buffer_name##_0[addr_name][2] = data_name.lane[2];\
	buffer_name##_0[addr_name][3] = data_name.lane[3];\
	buffer_name##_0[addr_name][4] = data_name.lane[4];\
	buffer_name##_0[addr_name][5] = data_name.lane[5];\
	buffer_name##_0[addr_name][6] = data_name.lane[6];\
	buffer_name##_0[addr_name][7] = data_name.lane[7];\
	buffer_name##_1[addr_name][0] = data_name.lane[8];\
	buffer_name##_1[addr_name][1] = data_name.lane[9];\
	buffer_name##_1[addr_name][2] = data_name.lane[10];\
	buffer_name##_1[addr_name][3] = data_name.lane[11];\
	buffer_name##_2[addr_name][0] = data_name.lane[12];\
	buffer_name##_2[addr_name][1] = data_name.lane[13];\
}

#define READ_BUF(buffer_name, addr_name, data_name)\
{\
	data_name.lane[0] = buffer_name##_0[addr_name][0];\
	data_name.lane[1] = buffer_name##_0[addr_name][1];\
	data_name.lane[2] = buffer_name##_0[addr_name][2];\
	data_name.lane[3] = buffer_name##_0[addr_name][3];\
	data_name.lane[4] = buffer_name##_0[addr_name][4];\
	data_name.lane[5] = buffer_name##_0[addr_name][5];\
	data_name.lane[6] = buffer_name##_0[addr_name][6];\
	data_name.lane[7] = buffer_name##_0[addr_name][7];\
	data_name.lane[8] = buffer_name##_1[addr_name][0];\
	data_name.lane[9] = buffer_name##_1[addr_name][1];\
	data_name.lane[10] = buffer_name##_1[addr_name][2];\
	data_name.lane[11] = buffer_name##_1[addr_name][3];\
	data_name.lane[12] = buffer_name##_2[addr_name][0];\
	data_name.lane[13] = buffer_name##_2[addr_name][1];\
}

#endif
