
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jittor as jt
import numpy as np
import argparse
import time
import random
import math

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix, SamplerRamdon

from dataset.exporter import Exporter
from models.skeleton import create_model

from PCT.misc.ops import knn_point, index_points
from utils.cheduler import LRScheduler

from models.loss import get_simplification_loss
from models.metrics import J2J
jt.flags.use_cuda = 1


def train(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    model = create_model(
        model_name= args.model_name,
        model_type=args.model_type
    )
    
    sampler = SamplerMix(num_samples=4096, vertex_samples=2048)
    
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    if args.lr_scheduler != 'none':
        scheduler = LRScheduler(
            optimizer=optimizer, 
            mode=args.lr_scheduler,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma,
            eta_min=args.lr_min,
            warmup_start_lr=args.lr_min,
            patience=args.lr_patience,
            factor=args.lr_factor,
            threshold=args.lr_threshold,
            cycles=args.lr_cycles,
            warmup_epochs=args.warmup_epochs,
            constant_epochs=args.constant_epochs
        )
    else:
        scheduler = None
    
    # Create loss function
    criterion = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_score = nn.CrossEntropyLoss()

    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
        random_pose=True
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size // 4,
            shuffle=False,
            sampler=sampler,
            transform=transform,
            random_pose=True
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            if batch_idx == len(train_loader) - 1 or batch_idx == 0:
                jt.sync_all()
                jt.gc()
            # Get data and labels
            vertices, joints, normals = data['vertices'], data['joints'], data['normals']
            
            outputs, skeleton_score, new_xyz , idx = model(vertices.permute(0, 2, 1), normals.permute(0, 2, 1))
            idx = knn_point(1, jt.Var(joints), jt.Var(vertices))
            skeleton_score_reshaped = skeleton_score.reshape(-1, skeleton_score.shape[-1])  # [B*8*4096, 52]
            idx_reshaped = idx.squeeze(-1).reshape(-1)  # [B*52]
            loss_score = criterion_score(skeleton_score_reshaped, idx_reshaped) 
            B, N, _ = vertices.shape
            _, J, _ = joints.shape
            outputs = outputs.reshape(B, -1)
            joints0 = joints.reshape(B, -1)
            loss1 = criterion(outputs, joints0)
            loss1_l1 = criterion_l1(outputs, joints0)

            j2j_loss = 0
            for i in range(outputs.shape[0]):
                temp = J2J(outputs[i].reshape(-1, 3), joints0[i].reshape(-1, 3))
                j2j_loss += temp / outputs.shape[0]

            if epoch < args.constant_epochs:
                loss =   loss_score
            else:
                loss = loss1 + loss1_l1 + j2j_loss+loss_score
            
            # Backward pass and optimize
            optimizer.zero_grad()   
            optimizer.backward(loss)
            optimizer.step()
            # Calculate statistics
            train_loss += loss.item()
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, "
                           f"Loss_l1: {loss1_l1.item():.4f}, J2J Loss: {j2j_loss.item():.4f}, Loss_score: {loss_score.item():.4f}")    
        current_lr = optimizer.lr
        if scheduler is not None:
            scheduler.step()
            log_message(f"Learning rate adjusted from {current_lr:.6f} to {optimizer.lr:.6f}")
        
        # Calculate epoch statistics
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")
        jt.sync_all()
        jt.gc()
        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            val_loss1 = 0.0
            val_loss1_l1 = 0.0
            val_loss_score = 0.0
            val_loss_score_l1 = 0.0
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints,  normals = data['vertices'], data['joints'], data['normals']
                B, N, _ = vertices.shape
                _, J, _ = joints.shape
                # Forward pass
                outputs, skeleton_score, new_xyz, idx = model(vertices.permute(0, 2, 1) , normals.permute(0, 2, 1))
                idx = knn_point(1, jt.Var(joints), jt.Var(vertices))
                loss_score = criterion_score(skeleton_score.reshape(-1, J), idx.squeeze(-1)) 

                _,idx = jt.topk(skeleton_score.permute(0,2,1), k= 50)
                
                skeleton_score = nn.softmax(skeleton_score, dim=2)
                vertices_topk = index_points(vertices, idx)
                meanp = jt.mean(vertices_topk, dim=2)

                joints0 = joints.reshape(joints.shape[0], -1)
                outputs = outputs.reshape(outputs.shape[0], -1)
                loss1 = criterion(outputs, joints0)
                loss1_l1 = criterion_l1(outputs, joints0)
                
                meanp1 = index_points(new_xyz, idx)
                meanp1 = jt.mean(meanp1, dim=2)
                j2j_loss = 0
                for i in range(outputs.shape[0]):
                    temp = J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3))
                    j2j_loss += temp / outputs.shape[0]
                
                loss = loss1 + loss1_l1 + j2j_loss+loss_score
                
                # export render results
                if batch_idx == show_id:
                    exporter = Exporter()
                    # export every joint's corresponding skinning
                    from dataset.format import parents
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_mean.png", joints=meanp[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].permute(1, 0).numpy())
                    exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/new_xyz.png", vertices=new_xyz[0].numpy())

                val_loss += loss.item()
                val_loss1 += loss1.item()
                val_loss1_l1 += loss1_l1.item()
                J2J_loss += j2j_loss.item()
                val_loss_score += loss_score.item()
                #val_loss_score += loss_score.item()
            # Calculate validation statistics
            val_loss /= len(val_loader)
            val_loss1 /= len(val_loader)
            val_loss1_l1 /= len(val_loader)
            J2J_loss /= len(val_loader)
            #val_loss_score /= len(val_loader)
            #val_loss_score_l1 /= len(val_loader)
            log_message(f"Validation Loss: {val_loss:.4f} Loss1: {val_loss1:.4f} Loss_l1: {val_loss1_l1:.4f} J2J Loss: {J2J_loss:.4f} Loss_score: {val_loss_score:.4f}")
            current_lr = optimizer.lr
            if scheduler is not None and args.lr_scheduler == 'plateau':
                scheduler.step(J2J_loss) 
                if current_lr != optimizer.lr:
                    log_message(f"Learning rate adjusted from {current_lr:.6f} to {optimizer.lr:.6f} based on validation performance")
            
            # Save best model
            if J2J_loss < best_loss:
                best_loss = J2J_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best models with loss {best_loss:.4f} to {model_path}")
            jt.sync_all()
            jt.gc()
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")

    return model, best_loss

def main():
    #     """Parse arguments and start training"""
    # parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # # Dataset parameters
    # parser.add_argument('--train_data_list', type=str, required=True,
    #                     help='Path to the training data list file')
    # parser.add_argument('--val_data_list', type=str, default='',
    #                     help='Path to the validation data list file')
    # parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/jittor-comp-human-main/data',
    #                     help='Root directory for the data files')
    
    # # Model parameters
    # parser.add_argument('--model_name', type=str, default='pct',
    #                     choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
    #                     help='Model architecture to use')
    # parser.add_argument('--model_type', type=str, default='standard',
    #                     choices=['standard', 'enhanced'],
    #                     help='Model type for skeleton model')
    # parser.add_argument('--pretrained_model', type=str, default='',
    #                     help='Path to pretrained model')
    
    # # Training parameters
    # parser.add_argument('--batch_size', type=int, default=16,
    #                     help='Batch size for training')
    # parser.add_argument('--epochs', type=int, default=100,
    #                     help='Number of training epochs')
    # parser.add_argument('--optimizer', type=str, default='adam',
    #                     choices=['sgd', 'adam'],
    #                     help='Optimizer to use')
    # parser.add_argument('--learning_rate', type=float, default=0.00001,
    #                     help='Initial learning rate')
    # parser.add_argument('--weight_decay', type=float, default=1e-4,
    #                     help='Weight decay (L2 penalty)')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='Momentum for SGD optimizer')
    
    # parser.add_argument('--lr_scheduler', type=str, default='step',
    #                     choices=['step', 'cosine', 'exp', 'plateau', 'cyclical', 'one_cycle', 'triangular', 'none'],
    #                     help='Learning rate scheduler')
    # parser.add_argument('--lr_step_size', type=int, default=20,
    #                     help='Step size for StepLR scheduler')
    # parser.add_argument('--lr_gamma', type=float, default=0.1,
    #                     help='Gamma for StepLR and ExponentialLR schedulers')
    # parser.add_argument('--lr_min', type=float, default=0.000001,
    #                     help='Minimum learning rate for CosineAnnealingLR scheduler')
    # parser.add_argument('--warmup_epochs', type=int, default=0,
    #                     help='Number of epochs for learning rate warmup')
    # parser.add_argument('--lr_cycles', type=int, default=1,
    #                     help='Number of cycles for cyclical scheduler')
    # parser.add_argument('--lr_patience', type=int, default=5,
    #                     help='Patience for plateau scheduler')
    # parser.add_argument('--lr_factor', type=float, default=0.5,
    #                     help='Factor for plateau scheduler')
    # parser.add_argument('--constant_epochs', type=int, default=0,
    #                     help='Number of epochs to keep learning rate constant at the beginning')
    
    # # Output parameters
    # parser.add_argument('--output_dir', type=str, default='output/skeleton',
    #                     help='Directory to save output files')
    # parser.add_argument('--print_freq', type=int, default=10,
    #                     help='Print frequency')
    # parser.add_argument('--save_freq', type=int, default=10,
    #                     help='Save frequency')
    # parser.add_argument('--val_freq', type=int, default=1,
    #                     help='Validation frequency')
    
    # args = parser.parse_args()
    
    # Start training
    class Args:
        # Dataset parameters
        train_data_list = 'data/train_list.txt'
        val_data_list = 'data/val_list.txt'
        data_root = 'data'
        # Model parameters
        model_name = 'pct'
        model_type = 'standard'
        pretrained_model = ''
        # Training parameters
        batch_size = 4
        epochs = 105
        optimizer = 'adam'
        learning_rate = 0.0001
        weight_decay = 1e-4
        momentum = 0.9
        lr_scheduler = 'cosine'  
        lr_step_size = 20
        lr_gamma = 0.5
        lr_min = 0.000001
        warmup_epochs = 0
        lr_cycles = 5
        lr_patience = 5
        lr_factor = 0.5
        lr_threshold = 0.01
        constant_epochs = 0
        # Output parameters
        output_dir = 'output/skeleton'
        print_freq = 10
        save_freq = 10
        val_freq = 1
    
    args = Args()
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()