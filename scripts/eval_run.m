function eval_run(det_root, iter, string)
% clear;

% The MATLAB code will use parfor for evaluation. Uncomment the following
% line and set the poolsize according to your need. Leave the line
% commented out if you want MATLAB to set the poolsize automatically.

pool_size = 1;


% image_set = 'test2015';
% iter = 150000;


% exp_name = 'rcnn_caffenet_union';  exp_dir = 'union';  prefix = 'rcnn_caffenet';  format = 'obj';

% exp_name = 'rcnn_caffenet_ho';  exp_dir = 'ho';  prefix = 'rcnn_caffenet';  format = 'obj';

% exp_name = 'rcnn_caffenet_ho_pfc_vec0';  exp_dir = 'ho_0';  prefix = 'rcnn_caffenet_pfc_vec';  format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pfc_vec1';  exp_dir = 'ho_1';  prefix = 'rcnn_caffenet_pfc_vec';  format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pfc_box0';  exp_dir = 'ho_0';  prefix = 'rcnn_caffenet_pfc_box';  format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pfc_box1';  exp_dir = 'ho_1';  prefix = 'rcnn_caffenet_pfc_box';  format = 'obj';

% exp_name = 'rcnn_caffenet_ho_pfc_ip0';    exp_dir = 'ho_0';  prefix = 'rcnn_caffenet_pfc_ip';    format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pfc_ip1';    exp_dir = 'ho_1';  prefix = 'rcnn_caffenet_pfc_ip';    format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pconv_ip0';  exp_dir = 'ho_0';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pconv_ip1';  exp_dir = 'ho_1';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';


% exp_name = 'rcnn_caffenet_ho_pconv_ip0';  exp_dir = 'ho_0';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';  score_blob = 'h';
% exp_name = 'rcnn_caffenet_ho_pconv_ip0';  exp_dir = 'ho_0';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';  score_blob = 'o';
% exp_name = 'rcnn_caffenet_ho_pconv_ip0';  exp_dir = 'ho_0';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';  score_blob = 'p';

% exp_name = 'rcnn_caffenet_ho_pconv_ip1';  exp_dir = 'ho_1';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';  score_blob = 'h';
% exp_name = 'rcnn_caffenet_ho_pconv_ip1';  exp_dir = 'ho_1';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';  score_blob = 'o';
% exp_name = 'rcnn_caffenet_ho_pconv_ip1';  exp_dir = 'ho_1';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';  score_blob = 'p';


% exp_name = 'rcnn_caffenet_ho_s';  exp_dir = 'ho_s';  prefix = 'rcnn_caffenet';  format = 'obj';

% exp_name = 'rcnn_caffenet_ho_pfc_ip0_s';    exp_dir = 'ho_0_s';  prefix = 'rcnn_caffenet_pfc_ip';    format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pfc_ip1_s';    exp_dir = 'ho_1_s';  prefix = 'rcnn_caffenet_pfc_ip';    format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pconv_ip0_s';  exp_dir = 'ho_0_s';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';
% exp_name = 'rcnn_caffenet_ho_pconv_ip1_s';  exp_dir = 'ho_1_s';  prefix = 'rcnn_caffenet_pconv_ip';  format = 'obj';


eval_mode = 'def';
eval_one(det_root, eval_mode, pool_size, iter, string);
eval_mode = 'ko';
eval_one(det_root, eval_mode, pool_size, iter, string);

end
