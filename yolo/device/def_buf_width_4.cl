#ifndef _DEF_BUF_WIDTH
#define _DEF_BUF_WIDTH

#define DEFINE_BUF(buffer_name, buffer_depth)\
	buffer_name##_0[buffer_depth][4]\
	

#define WRITE_BUF(buffer_name, addr_name, data_name)\
{\
	buffer_name##_0[addr_name][0] = data_name.lane[0];\
	buffer_name##_0[addr_name][1] = data_name.lane[1];\
	buffer_name##_0[addr_name][2] = data_name.lane[2];\
	buffer_name##_0[addr_name][3] = data_name.lane[3];\
}

#define READ_BUF(buffer_name, addr_name, data_name)\
{\
	data_name.lane[0] = buffer_name##_0[addr_name][0];\
	data_name.lane[1] = buffer_name##_0[addr_name][1];\
	data_name.lane[2] = buffer_name##_0[addr_name][2];\
	data_name.lane[3] = buffer_name##_0[addr_name][3];\
}

#endif
