import sys
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python [lane_num] [vec_size]'
	exit(1)
    lane = int(sys.argv[1])
    vec_num = int(sys.argv[2])
    all = lane*vec_num
    code_str = '#ifndef _PIPE_H\n'+'#define _PIPE_H\n'
    for i in range(0,all):
        code_str += 'pipe char data_ch' + str(i) + ' __attribute__((xcl_reqd_pipe_depth(32)));\n' +\
    	            'pipe char weight_ch' + str(i) + ' __attribute__((xcl_reqd_pipe_depth(32)));\n'
    for j in range(0,lane):
        code_str += 'pipe char bias_ch' + str(j) + ' __attribute__((xcl_reqd_pipe_depth(32)));\n'+\
    	            'pipe char conv_ch' + str(j) + ' __attribute__((xcl_reqd_pipe_depth(32)));\n'+\
    				'pipe char pool_ch' + str(j) + ' __attribute__((xcl_reqd_pipe_depth(32)));\n'+\
    				'pipe char bypass_ch' + str(j) + ' __attribute__((xcl_reqd_pipe_depth(32)));\n'
    code_str += '#define data_write_pipe_block(input_data)  '+\
                '{char temp[' + str(all) + '];\\\n'
    count = 0
    for i in range(0,lane):
    	for j in range(0,vec_num):
    		count = count + 1
    		if (count <= (all-1)):
    			code_str += '                                           temp['+str(count-1) +'] = input_data.lane[' + str(i) + '].data[' + str(j) + ']; \\\n' +\
    						'                                           write_pipe_block(data_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);\\\n'
    		else:
    			code_str += '                                           temp['+str(count-1) +'] = input_data.lane[' + str(i) + '].data[' + str(j) + ']; \\\n' +\
    						'                                           write_pipe_block(data_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);}\n'
    
    code_str += '#define data_read_pipe_block(input_data)  '+\
                '{char temp[' + str(all) + '];\\\n'
    count = 0
    for i in range(0,lane):
    	for j in range(0,vec_num):
    		count = count + 1
    		if (count <= (all-1)):
    			code_str += '                                           read_pipe_block(data_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);\\\n'+\
    			            '                                           input_data.lane[' + str(i) + '].data[' + str(j) + '] = temp['+str(count-1) +']; \\\n'
    
    		else:
    			code_str += '                                           read_pipe_block(data_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);\\\n'+\
    			            '                                           input_data.lane[' + str(i) + '].data[' + str(j) + '] = temp['+str(count-1) +'];} \n'
    
    
    code_str += '#define weight_write_pipe_block(input_data)  '+\
                '{char temp[' + str(all) + '];\\\n'
    count = 0
    for i in range(0,lane):
    	for j in range(0,vec_num):
    		count = count + 1
    		if (count <= (all-1)):
    			code_str += '                                           temp['+str(count-1) +'] = input_data.lane[' + str(i) + '].data[' + str(j) + ']; \\\n' +\
    						'                                           write_pipe_block(weight_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);\\\n'
    		else:
    			code_str += '                                           temp['+str(count-1) +'] = input_data.lane[' + str(i) + '].data[' + str(j) + ']; \\\n' +\
    						'                                           write_pipe_block(weight_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);}\n'
    
    code_str += '#define weight_read_pipe_block(input_data)  '+\
                '{char temp[' + str(all) + '];\\\n'
    count = 0
    for i in range(0,lane):
    	for j in range(0,vec_num):
    		count = count + 1
    		if (count <= (all-1)):
    			code_str += '                                           read_pipe_block(weight_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);\\\n'+\
    			            '                                           input_data.lane[' + str(i) + '].data[' + str(j) + '] = temp['+str(count-1) +']; \\\n'
    
    		else:
    			code_str += '                                           read_pipe_block(weight_ch'+str(i*vec_num+j)+', &temp['+str(count-1) +']);\\\n'+\
    			            '                                           input_data.lane[' + str(i) + '].data[' + str(j) + '] = temp['+str(count-1) +'];} \n'
    
    list_name = ['bias_ch','conv_ch','pool_ch','bypass_ch']
    for n in range(0,4):
    	code_str += '#define '+ list_name[n]+'_write_pipe_block(input_data)  '+\
    				'{char temp[' + str(lane) + '];\\\n'
    	count=0
    	for i in range(0,lane):
    		count = count + 1
    		if (count <= (lane-1)):
    			code_str += '                                           temp['+str(count-1) +'] = input_data.lane[' + str(i) + ']; \\\n' +\
    						'                                           write_pipe_block('+list_name[n]+str(i)+', &temp['+str(count-1) +']);\\\n'
    		else:
    			code_str += '                                           temp['+str(count-1) +'] = input_data.lane[' + str(i) + ']; \\\n' +\
    						'                                           write_pipe_block('+list_name[n]+str(i)+', &temp['+str(count-1) +']);}\n'
    	code_str += '#define '+ list_name[n]+'_read_pipe_block(input_data)  '+\
    				'{char temp[' + str(lane) + '];\\\n'
    	count=0
    	for i in range(0,lane):
    		count = count + 1
    		if (count <= (lane-1)):
    			code_str += '                                           read_pipe_block('+list_name[n]+str(i)+', &temp['+str(count-1) +']);\\\n' +\
    			            '                                           input_data.lane[' + str(i) + '] = temp['+str(count-1) +']; \\\n'
    
    		else:
    			code_str += '                                           read_pipe_block('+list_name[n]+str(i)+', &temp['+str(count-1) +']);\\\n' +\
    			            '                                           input_data.lane[' + str(i) + '] = temp['+str(count-1) +'];} \n'
    code_str += '#endif\n'
    fd = open('pipe.cl', 'w')
    fd.write(code_str)
    fd.close()
