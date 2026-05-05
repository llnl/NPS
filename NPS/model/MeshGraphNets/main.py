# Lint as: python3
# pytorch port
# ============================================================================

import time
import pickle
import os, sys; sys.path.append(os.path.join(sys.path[0], '..'))
import numpy as np
import torch
import torch.nn as nn

from MeshGraphNets.option import args
# tf.logging.set_verbosity(tf.logging.ERROR)
# from meshgraphnets import cfd_eval
# from meshgraphnets import amr_eval
from MeshGraphNets import NPS_model
from MeshGraphNets import base_mp_gnn, common
from NPS.utility import make_optimizer, make_scheduler, count_parameters
# from meshgraphnets import dataset
# import horovod.tensorflow as hvd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# PARAMETERS = {
#     'NPS': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
#                 size=2, batch=2, model=NPS_model, evaluator=cfd_eval),
# }

def train_on_batch(input, target, model, optimizer, criterion,teacher_forcing_ratio,
      input_length=1, target_length=1, RNN=False):
    model.train()
    optimizer.zero_grad()
    # # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    # input_length  = input.size(1)
    # target_length = target.size(1)
    assert RNN or (input_length==1), ValueError('Set input_length=1 when disabling RNN')
    out = model(input)
    loss = criterion(out, target.reshape(-1,1))
    # loss = 0
    # for ei in range(input_length-1): 
    #     output,_,_ = model(input[:,ei,:,:,:], (ei==0) )
    #     loss += criterion(output,input[:,ei+1,:,:,:])

    # decoder_input = input[:,-1,:,:,:] # first decoder input = last image of input sequence
    
    # use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False 
    # for di in range(target_length):
    #     output_image,_,_ = encoder(decoder_input, (not RNN) or ((input_length==1) and (di==0)))
    #     target = target[:,di,:,:,:]
    #     loss += criterion(output_image,target)
    #     if use_teacher_forcing:
    #         decoder_input = target # Teacher forcing    
    #     else:
    #         decoder_input = output_image

    # # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,args.phy_ker,args.phy_ker)
    # for b in range(0,encoder.phycell.cell_list[0].input_dim):
    #     filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,args.phy_ker,args.phy_ker)
    #     m = k2m(filters.double()) 
    #     m  = m.float()   
    #     loss += criterion(m, constraints) # constrains is a precomputed matrix   
    loss.backward()
    optimizer.step()
    return loss.item() / target_length

def train(model, loader, loader_valid, args, device='cpu', checkpoint=None, mesher=None, nepochs=1000,
      print_every=10, eval_every=10, noise=0,
      input_length=1, target_length=1, RNN=False):
    train_losses = []

    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)
    criterion = loss_fn
    epoch_start = 0
    best_loss = float('inf')
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"restored optimizer from {args.dir}/model.pth")
        epoch_start = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
    except:
        pass

    re_graph=True
    # for epoch in range(epoch_start, nepochs):
    #     t0 = time.time()
    #     loss_epoch = 0
    #     teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 

    epoch = epoch_start-1
    while True:
        if epoch > nepochs: break
        for dat in loader:
            epoch+=1
            if epoch > nepochs: break
            t0 = time.time()
            teacher_forcing_ratio = np.maximum(0 , 1 - epoch * 0.003) 
            # print(f'---epoch {epoch} {i}')
            input = dat[1].to(device)
            target = dat[2].to(device)
            if noise > 0:
                input += noise* torch.randn_like(input)
            # if re_graph:
            g0 = common.array2graph(input, args.dim, periodic=args.periodic, device=device)
                # re_graph=False
            loss = train_on_batch(g0, target, model, optimizer, criterion, teacher_forcing_ratio, RNN=RNN)
            # loss_epoch += loss
                      
            # train_losses.append(loss_epoch)        
            if (epoch) % print_every == 0:
                print(f'epoch {epoch:8d} loss       {loss:8.3g} time {time.time()-t0:7.3g}')
                
            if (epoch) % eval_every == 0:
                t0 = time.time()
                loss_valid, mae = evaluate(model, loader_valid, mesher=mesher, n_out=args.n_out_frame, RNN=RNN)
                print(f'epoch {epoch:8d} valid_loss {loss_valid:8.3g} time {time.time()-t0:7.3g}')
                scheduler.step(loss_valid)
                if loss_valid < best_loss:
                    best_loss = loss_valid
                    print(f'New best validation loss saving step {epoch}')
                    torch.save({
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},f'{args.dir}/model.pth')
    return train_losses

    # ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
    # if FLAGS.cache:
    #     ds = ds.cache()
    # if FLAGS.randommesh:
    #     ds = ds.map(dataset.augment_by_randommesh, periodic=FLAGS.periodic)
    # ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
    # ds = dataset.split(ds)
    # if FLAGS.rotate:
    #     ds = ds.map(dataset.augment_by_rotation(FLAGS.dim, FLAGS.rotate))
    # if mesher is not None:
    #     ds = dataset.remesh(ds, mesher, random_translate=False)
    # ds = dataset.add_training_noise(ds, noise_field=params['field'],
    #                                                                     noise_scale=params['noise'] if FLAGS.noise<0 else FLAGS.noise,
    #                                                                     noise_gamma=params['gamma'])
    # ds = dataset.batch_dataset(ds, FLAGS.batch)
    # # inputs = tf.data.make_one_shot_iterator(ds).get_next()
    # ds_iterator = tf.data.make_initializable_iterator(ds)
    # inputs = ds_iterator.get_next()

    # loss_op = model.loss(inputs)
    # global_step = tf.train.create_global_step()
    # lr = tf.train.exponential_decay(learning_rate=FLAGS.lr,
    #                                                                 global_step=global_step,
    #                                                                 decay_steps=FLAGS.lr_decay,
    #                                                                 decay_rate=0.1) + 1e-6
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # train_op = optimizer.minimize(loss_op, global_step=global_step)
    # # Don't train for the first few steps, just accumulate normalization stats
    # train_op = tf.cond(tf.less(global_step, 1000),
    #                                      lambda: tf.group(tf.assign_add(global_step, 1)),
    #                                      lambda: tf.group(train_op))
    # valid_op, nvalid = evaluator(model, params, 'valid', None, mesher, return_valid=True)

    # saver=tf.train.Saver(max_to_keep=5 if FLAGS.keep_ckpt<=0 else FLAGS.keep_ckpt)
    # with tf.train.MonitoredTrainingSession(
    #         hooks=[tf.train.StopAtStepHook(last_step=args.n_training_steps)],
    #         checkpoint_dir=FLAGS.checkpoint_dir,
    #         scaffold=None,save_checkpoint_steps=None,#tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=20)) if FLAGS.keep_ckpt>0 else None,
    #         save_checkpoint_secs=None) as sess:
    #     best_valid = 1.0e99

    #     sess.run(ds_iterator.initializer)
    #     while not sess.should_stop():
    #         _, step, loss = sess.run([train_op, global_step, loss_op])
    #         if step % 1000 == 0:
    #             logging.info('Step %d: Loss %g', step, loss)
    #         if step % FLAGS.valid_freq == 0:
    #             logging.info(f'Validating {nvalid}')
    #             validation_err = [list(sess.run([valid_op])[0].values()) for _ in range(nvalid)]
    #             valid_err_np = np.mean(validation_err,0)
    #             logging.info(f' Step {step} validation err {valid_err_np}')
    #             if valid_err_np[-1] < best_valid:
    #                 best_valid = valid_err_np[-1]
    #                 saver.save(sess._sess._sess._sess._sess, FLAGS.checkpoint_dir+'/model.ckpt', global_step=step)
    #     logging.info('Training complete.')
    # evaluate(model, params, 'valid', None, mesher)


# from NPS.utility import get_gpu_memory_map
def evaluate(model, loader, rollout=None, mesher=None, n_out=None, n_rollout=-1, RNN=False):
    """Run a model rollout trajectory."""
    model.eval()
    nvalid = len(loader)
    rollout_traj = []
    loss_list = []
    with torch.no_grad():
        for i, dat in enumerate(loader):
            print(time.time())
            if n_rollout>=0 and i >= n_rollout: break
            input = dat[1].to(device)
            target = dat[2].to(device)
            g0 = common.array2graph(input, args.dim, periodic=args.periodic, device=device)
            # print(f'debug graph {g0}')
            for _ in range(n_out):
                out = model(g0)
                g0['x'] = out
                loss = loss_fn(out, target.reshape(-1,1))
                rollout_traj.append(out.detach().cpu().numpy())
                loss_list.append(loss.item())
    trajectories = np.stack(rollout_traj)
    if rollout:
        with open(rollout, 'wb') as fp:
            pickle.dump(trajectories, fp)
        np.save(f'{rollout}.npy', trajectories)
    # print(loss_list)
    return np.mean(loss_list), 0
    # # ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
    # # if return_valid:
    # #     ds = ds.repeat(None)
    # # inputs = tf.data.make_one_shot_iterator(ds).get_next()
    # # scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs, mesher=mesher, num_steps=predict_steps)
    # if return_valid:
    #     return scalar_op, nvalid
    # try:
    #     tf.train.create_global_step()
    # except:
    #     pass

    # with tf.train.MonitoredTrainingSession(
    #         checkpoint_dir=FLAGS.checkpoint_dir,
    #         save_checkpoint_secs=None,
    #         save_checkpoint_steps=None) as sess:
    #     trajectories = []
    #     scalars = []
    #     mse_list = []
    #     for traj_idx in range(FLAGS.num_rollouts):
    #         logging.info('Rollout trajectory %d', traj_idx)
    #         scalar_data, traj_data = sess.run([scalar_op, traj_ops])
    #         trajectories.append(traj_data)
    #         # error = traj_data['pred_velocity'] - traj_data['gt_velocity']
    #         # mse_list.append((error**2).mean(axis=1))
    #         if predict_steps is None:
    #             error = [np.mean((traj_data['pred_velocity'][i] - traj_data['gt_velocity'][i])**2, axis=0) for i in range(len(traj_data['pred_velocity']))]
    #             mse_list.append(error)
    #             scalars.append(scalar_data)
    #     logging.info(f'Rollout trajectory {FLAGS.num_rollouts} total done')
    #     if predict_steps is None:
    #         for key in scalars[0]:
    #             logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    #         print(f'RMSE   total {np.sqrt(np.mean(mse_list))}')
    #         print(f' per_channel {np.sqrt(np.mean(mse_list,axis=(0,1)))}')
    #         print(f'    per_step {np.sqrt(np.mean(mse_list,axis=(0,2)))}')
    #         print(f'    per_traj {np.sqrt(np.mean(mse_list,axis=(1,2)))}')
    #     # variables_names = [v.name for v in tf.trainable_variables()]
    #     # values = sess.run(tf.trainable_variables())
    #     # for k, v in zip(variables_names, values):
    #     #     print( "Variable: ", k, v.shape, v)


if __name__ == '__main__':
    os.makedirs(args.dir, exist_ok=True)
    with open(f"{args.dir}/command.txt", "a") as f:
        f.write(' '.join(sys.argv) + '\n')
    loss_fn = nn.MSELoss()

#################### Data ####################
    if args.dataset == 'longclip':
        import NPS.data
        from collections import namedtuple
        if args.ds_pred:
            file_pred = args.ds_pred
        else:
            file_pred = {"eval":"valid","train":"valid","predict":"test"}[args.mode]
            file_pred = f'{args.data}/{file_pred}.npy'
        data_args = {'datatype_train':'longclip', 'datatype_test':'longclip',
        'file_train':f'{args.data}/train.npy', 'file_test':file_pred,
        'minibatch_size':args.batch, 'cpu':False, 'n_threads':1, 'test_only':False,
        'dim':args.dim, 'data_slice':'', 'data_filter':'', 'data_preprocess':'',
        'channel_first':True, 'space_CG':False, 'frame_shape':args.frame_shape, 'time_CG':1,
        'total_length_test':args.n_in_frame+(args.n_out_frame if args.n_out_pred<0 else args.n_out_pred),
        'clip_step_test':args.clip_step if args.clip_step_test<0 else args.clip_step_test,
        'total_length':args.n_in_frame+args.n_out_frame, 'frame_step':1, 'clip_step':args.clip_step, 'i_in_out':True,'n_in':args.n_in_frame}
        data_args = namedtuple('Data_args_init', data_args.keys())(*data_args.values())
        print(f'debug ds_pred {args.ds_pred}, args {data_args}')
        dataset = NPS.data.Data(data_args)
        loader_train = torch.utils.data.DataLoader(dataset=dataset.train, batch_size=args.batch, shuffle=True, num_workers=0)
        loader_valid =  torch.utils.data.DataLoader(dataset=dataset.test, batch_size=args.batch, shuffle=False, num_workers=0)
    elif args.dataset.startswith('hubbard1band'):
        from NPS.data.hubbard1band import load_hubbard1band
        from torch_geometric.loader import DataLoader
        from sklearn.model_selection import train_test_split
        # print(f'debug data', args.data, load_hubbard1band(args.data, cache_data=args.cache))
        ds_train, ds_valid = train_test_split(load_hubbard1band(args.data, cache_data=args.cache, filter=args.dataset[12:]), train_size=args.train_split, random_state=12345)
        loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True)
        loader_valid = DataLoader(ds_valid, batch_size=args.batch, shuffle=False)

#################### Model ####################
    if args.core == 'graph_gradient_model':
        from MeshGraphNets import graph_gradient_model
        core_model_type = graph_gradient_model.GNGradientNet
    elif args.core == 'diffusion':
        from MeshGraphNets import diffusion_model
        core_model_type = diffusion_model.EncodeProcessDecodeDiffusion
    else:
        core_model_type = base_mp_gnn.EncodeProcessDecode
    learned_model = core_model_type(
            input_size_node=args.nfeat_in+(0 if args.model == 'GNN' else common.NodeType.SIZE), input_size_edge=args.dim+1,
            output_size=args.nfeat_out + args.nfeat_out_global,
            activation=args.mlp_activation,
            latent_size_node=args.nfeat_latent_node, latent_size_edge=args.nfeat_latent_edge,
            num_layers=args.nlayer_mlp,
            message_passing_steps=args.n_mpassing)
    if args.model in ['NPS']:
        model = NPS_model.BaseNPSGNNModel(learned_model, 
            dim=args.dim, periodic=args.periodic, nfeat_in=args.nfeat_in,
            nfeat_out=args.nfeat_out, unique_op=args.unique_op)
        trainer = train
    elif args.model == 'GNN':
        from MeshGraphNets import GNN 
        model = GNN.GNN(learned_model, 
            nfeat_in=args.nfeat_in,
            nfeat_out=args.nfeat_out, nfeat_out_global=args.nfeat_out_global)
        trainer = GNN.train
    else:
        raise ValueError(f'unknown model {args.model}')
    model.to(device)
    print(f'model {count_parameters(model)} parameters\n', model)
    try:
        from torchinfo import summary
        summary(model)
    except:
        pass
    checkpoint = None
    try:
        checkpoint = torch.load(f'{args.dir}/model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"restored model from {args.dir}/model.pth")
    except:
        pass
    if args.amr_N1 > 1:
        from meshgraphnets import amr
        print('''************* WARNING *************
        The present AMR implementation assumes an input field on a cubic grid of size amr_N
        ordered naturally. Make sure your dataset follows this convention''')
        mesher = amr.amr_state_variables(args.dim, [args.amr_N]*args.dim,
            [args.amr_N//args.amr_N1]*args.dim,
            torch.zeros([args.amr_N**args.dim,1],dtype=torch.float32),
            refine_threshold=args.amr_threshold, buffer=args.amr_buffer, eval_freq=args.amr_eval_freq)
        params['evaluator'] = amr_eval
    else:
        mesher = None
    # if args.evaluator is not None:
    #     if FLAGS.evaluator == 'cfd_eval':
    #         params['evaluator'] = cfd_eval
    #     elif FLAGS.evaluator == 'cloth_eval':
    #         params['evaluator'] = cloth_eval
    #     elif FLAGS.evaluator == 'amr_eval':
    #         params['evaluator'] = amr_eval
    #     elif FLAGS.evaluator == 'featureevolve_eval':
    #         from meshgraphnets import featureevolve_eval
    #         params['evaluator'] = featureevolve_eval
    #     elif FLAGS.evaluator == 'amr_featureevolve_eval':
    #         from meshgraphnets import amr_featureevolve_eval
    #         params['evaluator'] = amr_featureevolve_eval
    #     else:
    #         raise ValueError(f'ERROR: unknown evaluator {FLAGS.evaluator}')

#################### Job ####################
    if args.mode == 'train':
        trainer(model, loader_train, loader_valid, args, device, checkpoint, 
            mesher=mesher, nepochs=args.nepoch, 
            print_every=args.print_freq, eval_every=args.valid_freq,
            noise=args.noise, RNN=args.RNN)
    elif args.mode == 'eval':
        evaluate(model, loader_valid, args.rollout, mesher)
    elif args.mode == 'predict':
        # print('loader', pred_loader, next(iter(pred_loader)))
        # print('ds', pred_loader.dataset, pred_loader.dataset[0])
        evaluate(model, loader_valid, args.rollout, mesher, 
          n_out=args.n_out_pred, n_rollout=args.n_rollout,
          RNN=args.RNN)
    else:
        raise ValueError(f'Unknown job mode {args.mode}')

