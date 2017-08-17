## Where to get

You can download the fixed-point CNN weights, input test vectors and output golden reference data for AlexNet/VGG-16 from the following links to run and evaluate the project.

* Baidu's Cloud Drive:
```
http://pan.baidu.com/s/1jIl6qkm
```
* GoogleDrive:
```
https://drive.google.com/open?id=0B3srpZY5rHcASEhSSmh3Tm5LclU
```

## Fixed-Point CNN models

The weights are quantilized with 8-bit precisions, and the model quantization is performed such that there is only 1% loss in top-1/5 accuracy for all models. In quantization, we assume the fixed-point weight is presented as *N * 2^-m*, where *N* is a fixed-point integer with *n-bit* word length, and *m* denotes the fractional bits of the quantized weight. Therefore, we use the pair of integers *(n,m)* as the quantization parameters. In the following tables, we report the quantization parameters used in this project for different CNN models.

* AlexNet(CaffeNet)

<table>
   <tr>
      <td>LayerName</td>
      <td>Input</td>
      <td>Output</td>
      <td>Weight</td>
   </tr>
   <tr>
      <td>conv1</td>
      <td>8,0</td>
      <td>8,-4</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu1</td>
      <td>8.-4</td>
      <td>8,-4</td>
      <td></td>
   </tr>
   <tr>
      <td>lrn1</td>
      <td>8,-4</td>
      <td>8,0</td>
      <td></td>
   </tr>
   <tr>
      <td>pool1</td>
      <td>8,0</td>
      <td>8,0</td>
      <td></td>
   </tr>
   <tr>
      <td>conv2</td>
      <td>8,0</td>
      <td>8,-2</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu2</td>
      <td>8,-2</td>
      <td>8,-2</td>
      <td></td>
   </tr>
   <tr>
      <td>lrn2</td>
      <td>8,-2</td>
      <td>8,0</td>
      <td></td>
   </tr>
   <tr>
      <td>pool2</td>
      <td>8,0</td>
      <td>8,0</td>
      <td></td>
   </tr>
   <tr>
      <td>conv3</td>
      <td>8,0</td>
      <td>8,-1</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu3</td>
      <td>8,-1</td>
      <td>8,-1</td>
      <td></td>
   </tr>
   <tr>
      <td>conv4</td>
      <td>8,-1</td>
      <td>8,-1</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu4</td>
      <td>8,-1</td>
      <td>8,-1</td>
      <td></td>
   </tr>
   <tr>
      <td>conv5</td>
      <td>8,-1</td>
      <td>8,-1</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu5</td>
      <td>8,-1</td>
      <td>8,-1</td>
      <td></td>
   </tr>
   <tr>
      <td>pool5</td>
      <td>8,-1</td>
      <td>8,-1</td>
      <td></td>
   </tr>
   <tr>
      <td>fc6</td>
      <td>8,-1</td>
      <td>8,0</td>
      <td>8,11</td>
   </tr>
   <tr>
      <td>relu6</td>
      <td>8,0</td>
      <td>8,0</td>
      <td></td>
   </tr>
   <tr>
      <td>drop6</td>
      <td>8,0</td>
      <td>8,0</td>
      <td></td>
   </tr>
   <tr>
      <td>fc7</td>
      <td>8,0</td>
      <td>8,2</td>
      <td>8,10</td>
   </tr>
   <tr>
      <td>relu7</td>
      <td>8,2</td>
      <td>8,2</td>
      <td></td>
   </tr>
   <tr>
      <td>drop7</td>
      <td>8,2</td>
      <td>8,2</td>
      <td></td>
   </tr>
   <tr>
      <td>fc8</td>
      <td>8,2</td>
      <td>8,2</td>
      <td>8,10</td>
   </tr>
</table>


* VGG-16

<table>
   <tr>
      <td>LayerName</td>
      <td>Input</td>
      <td>Output</td>
      <td>Weight</td>
   </tr>
   <tr>
      <td>conv1_1</td>
      <td>8,0</td>
      <td>8,-2</td>
      <td>8,7</td>
   </tr>
   <tr>
      <td>relu1_1</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv1_2</td>
      <td>8,-2</td>
      <td>8,-5</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu1_2</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv2_1</td>
      <td>8,-5</td>
      <td>8,-5</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu2_1</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv2_2</td>
      <td>8,-5</td>
      <td>8,-6</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu2_2</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv3_1</td>
      <td>8,-6</td>
      <td>8,-7</td>
      <td>8,7</td>
   </tr>
   <tr>
      <td>relu3_1</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv3_2</td>
      <td>8,-7</td>
      <td>8,-7</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu3_2</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv3_3</td>
      <td>8,-7</td>
      <td>8,-7</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu3_3</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv4_1</td>
      <td>8,-7</td>
      <td>8,-6</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu4_1</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv4_2</td>
      <td>8,-6</td>
      <td>8,-5</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu4_2</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv4_3</td>
      <td>8,-5</td>
      <td>8,-5</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu4_3</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv5_1</td>
      <td>8,-5</td>
      <td>8,-4</td>
      <td>8,9</td>
   </tr>
   <tr>
      <td>relu5_1</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv5_2</td>
      <td>8,-4</td>
      <td>8,-3</td>
      <td>8,9</td>
   </tr>
   <tr>
      <td>relu5_2</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>conv5_3</td>
      <td>8,-3</td>
      <td>8,-2</td>
      <td>8,8</td>
   </tr>
   <tr>
      <td>relu5_3</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>fc6</td>
      <td>8,-2</td>
      <td>8,0</td>
      <td>4,8</td>
   </tr>
   <tr>
      <td>fc7</td>
      <td>8,0</td>
      <td>8,2</td>
      <td>4,7</td>
   </tr>
   <tr>
      <td>fc8</td>
      <td>8,2</td>
      <td>8,2</td>
      <td>4,7</td>
   </tr>
</table>

## How to prepare the CNN models for PipeCNN

* Install [Caffe](http://caffe.berkeleyvision.org/) and use the following matlab script to extract the CNN model (assuming we are dealing with caffenet)
```
caffe.set_mode_cpu();

model = './models/bvlc_reference_caffenet/deploy.prototxt';
weights = './models/bvlc_reference_caffenet.caffemodel';

net = caffe.Net(model, weights, 'test');

netparams = {{net.params('conv1',1).get_data(),net.params('conv1',2).get_data()}, ...
			{net.params('conv2',1).get_data(),net.params('conv2',2).get_data()}, ...
			{net.params('conv3',1).get_data(),net.params('conv3',2).get_data()}, ...
			{net.params('conv4',1).get_data(),net.params('conv4',2).get_data()}, ...
			{net.params('conv5',1).get_data(),net.params('conv5',2).get_data()}, ...
			{net.params('fc6',1).get_data(),net.params('fc6',2).get_data()}, ...
			{net.params('fc7',1).get_data(),net.params('fc7',2).get_data()}, ...
			{net.params('fc8',1).get_data(),net.params('fc8',2).get_data()}};
```

* Quantize the extracted mode (stored in netparams) to fixed-point weight and bias. Matlab's Fixed-point toolbox should be installed. The word length and fractional bit length are set according to the tables in previous sections.
```
WeightWidth    = [ 8;  8;  8;  8;  8;  8;  8;  8];
WeightFrac     = [ 8;  8;  8;  8;  8; 11; 10; 10];

MathType   = fimath('RoundingMethod', 'Nearest', 'OverflowAction', 'Saturate', 'ProductMode', 'FullPrecision', 'SumMode', 'FullPrecision');

for i=1:8
	WeightType{i}  = numerictype('Signed',1, 'WordLength', WeightWidth(i), 'FractionLength', WeightFrac(i));
	weight{i}  = fi(netParams.netparams{i}{1}, WeightType{i}, MathType);
	bias{i}    = fi(netParams.netparams{i}{2}, WeightType{i}, MathType);
end

```

* Combine and store the weights and bias as a single binary file.
```
fid = fopen('weights.dat', 'w');
for i=1:8
    fwrite(fid, storedInteger(weight{i}), 'int8');
    fwrite(fid, storedInteger(bias{i}), 'int8');
end
fclose(fid);
```


## Notes:

Remember to change the corresponding paths in "main.cpp" before running the project.