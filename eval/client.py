import sys
import os
import subprocess
import json
import time
import signal
import faulthandler
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.join(os.getcwd(), 'script')


def diagnostic_q(env):
    return np.asarray(list(env.robot.get_left_arm_jointState()) + list(env.robot.get_right_arm_jointState()), dtype=float)


def diagnostic_write(path, row):
    row['wall_time'] = time.time()
    with open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def diagnostic_execute(env, action, path, chunk_id, action_index):
    before = diagnostic_q(env)
    physical_before = np.asarray(env.robot.get_left_arm_real_jointState()[:6] + env.robot.get_right_arm_real_jointState()[:6], dtype=float)
    start_step = int(env.take_action_cnt)
    started = time.monotonic()
    env.take_action(action)
    after = diagnostic_q(env)
    physical_after = np.asarray(env.robot.get_left_arm_real_jointState()[:6] + env.robot.get_right_arm_real_jointState()[:6], dtype=float)
    row = {'kind':'action', 'chunk_id':chunk_id, 'action_index':action_index,
           'step_before':start_step, 'step_after':int(env.take_action_cnt),
           'command_q':np.asarray(action).tolist(), 'drive_target_before':before.tolist(),
           'drive_target_after':after.tolist(), 'physical_arm_q_before':physical_before.tolist(),
           'physical_arm_q_after':physical_after.tolist(), 'execution_seconds':time.monotonic()-started,
           'eval_success':bool(env.eval_success), 'stage_success_tag':bool(getattr(env,'stage_success_tag',False))}
    if env.eval_success:
        contacts=[]
        for contact in env.scene.get_contacts():
            contacts.append({'bodies':[b.entity.name for b in contact.bodies],
                             'positions':[np.asarray(p.position).tolist() for p in contact.points]})
        row['success_contacts']=contacts
        if hasattr(env,'bell'):
            row['bell_top']=np.asarray(env.bell.get_contact_point(0)[:3]).tolist()
    diagnostic_write(path,row)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None
    video_fps = str(usr_args.get("video_fps", 10))

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if usr_args.get("output_dir"):
        save_dir = Path(usr_args["output_dir"]) / task_name
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 命令行 --eval_video_log 优先于 YAML 配置
    if "eval_video_log" in usr_args:
        args["eval_video_log"] = usr_args["eval_video_log"]

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    args['baseline_save_dir'] = str(save_dir)

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = int(os.environ.get("ROBOTWIN_TEST_EPISODES", "100"))
    if test_num < 1:
        raise ValueError("ROBOTWIN_TEST_EPISODES must be positive")
    topk = 1

    # model = get_model(usr_args)
    # from IPython import embed;embed()
    from script.deploy.websocket_client_policy import WebsocketClientPolicy
    model = WebsocketClientPolicy(port=usr_args['port'])

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   video_fps=video_fps,
                                   instruction_type=instruction_type,
                                   usr_args=usr_args)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    result = {'task':task_name, 'setting':task_config, 'attempts':test_num,
              'successes':int(suc_num), 'success_rate':float(suc_num/test_num), 'complete':True}
    result_path = save_dir/'result.json'
    result_path.with_suffix('.tmp').write_text(json.dumps(result, indent=2))
    result_path.with_suffix('.tmp').replace(result_path)
    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                video_fps="10",
                instruction_type=None,
                usr_args = None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    recovery_dir = Path(args['baseline_save_dir'])
    ledger = recovery_dir/'episodes.jsonl'
    recovered = [json.loads(line) for line in ledger.read_text().splitlines() if line] if ledger.exists() else []
    if len({x['seed'] for x in recovered}) != len(recovered):
        raise ValueError('Duplicate recovered seeds')
    TASK_ENV.suc = sum(bool(x['success']) for x in recovered)
    TASK_ENV.test_num = len(recovered)
    last_heartbeat = [0.0]
    def heartbeat(stage, force=False):
        now=time.time()
        if not force and now-last_heartbeat[0]<1: return
        payload={'time':now,'stage':stage,'seed':now_seed,'completed':TASK_ENV.test_num,
                 'step':int(getattr(TASK_ENV,'take_action_cnt',0)),'pid':os.getpid()}
        tmp=recovery_dir/'heartbeat.tmp'
        tmp.write_text(json.dumps(payload));tmp.replace(recovery_dir/'heartbeat.json')
        last_heartbeat[0]=now
    pending_path=recovery_dir/'pending_episode.json'
    pending=json.loads(pending_path.read_text()) if pending_path.exists() else None

    now_id = len(recovered)
    succ_seed = len(recovered)
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    # eval_func = eval_function_decorator(policy_name, "eval")
    # reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = recovered[-1]["seed"]+1 if recovered else st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        if now_seed - st_seed >= max(1000, test_num * 100):
            raise RuntimeError('Expert feasibility seed budget exceeded; not a policy failure')
        heartbeat('expert_setup',True)
        episode_started = time.monotonic()
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                heartbeat('expert_play',True)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                # stack_trace = traceback.format_exc()
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                with open(Path(args['baseline_save_dir'])/'setup_errors.jsonl', 'a') as f:
                    f.write(json.dumps({'seed': now_seed-1, 'error': repr(e)})+'\n')
                print(f"error occurs ! {e}")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        import random
        random.seed(now_seed)
        np.random.seed(now_seed)
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        if pending and pending['seed']==now_seed:
            instruction=pending['instruction']
        pending_path.write_text(json.dumps({'seed':now_seed,'instruction':instruction}))
        heartbeat('policy_reset',True)
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    video_fps,
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    "-threads",
                    "2",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        diag_path = recovery_dir / f'diagnostic_episode{TASK_ENV.test_num}_seed{now_seed}.jsonl'
        chunk_id = 0
        diagnostic_write(diag_path, {'kind':'episode_start', 'seed':now_seed, 'episode':TASK_ENV.test_num,
                         'instruction':instruction, 'joint_order':'left6,left_gripper,right6,right_gripper',
                         'units':'arm radians; gripper environment coordinates',
                         'sampling':'before/after each policy action, not every physics substep',
                         'initial_drive_target':diagnostic_q(TASK_ENV).tolist(), 'diagnostic_schema':2, 'physical_arm_order':'left6,right6'})
        succ = False
        path_to_pi_model = None
        if usr_args is not None and "new_ckpt_path" in usr_args:
            path_to_pi_model = usr_args['new_ckpt_path']
        ret = model.infer(dict(reset = True, robo_name=usr_args['robo_name'], path_to_pi_model=path_to_pi_model, eval_seed=now_seed))
        
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim and not succ:
            heartbeat('get_obs')
            observation = TASK_ENV.get_obs()
            # from IPython import embed;embed()
            
            formatted_observation = {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                
                "observation.state": observation["joint_action"]["vector"],
                "task": TASK_ENV.get_instruction(),
            }

            # print(formatted_observation["observation.images.cam_high"].shape)
            # ========= debug image =========
            # import torch, torchvision
            # from PIL import Image
            # imgs_np = []
            # for k in ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']:
            #     arr = formatted_observation['image'][k].squeeze(0)  # (H,W,3), uint8
            #     print(arr.max(), arr.min())
            #     imgs_np.append(arr)

            # concat = np.concatenate(imgs_np, axis=1)
            # if TASK_ENV.take_action_cnt % 25 ==0:
            #     Image.fromarray(concat).save(f'combined_{TASK_ENV.take_action_cnt}.png')
            # ========= debug image =========

            # from IPython import embed;embed()
            heartbeat('model_infer',True)
            ret = model.infer(formatted_observation) #(TASK_ENV, model, observation)
            action, latency = ret['action'], ret['server_timing']
            if not np.isfinite(action).all() or action.shape[-1] != 14:
                raise RuntimeError(f'Invalid action: shape={action.shape}, finite={np.isfinite(action).all()}')
            diagnostic_write(diag_path, {'kind':'chunk', 'chunk_id':chunk_id,
                             'step':int(TASK_ENV.take_action_cnt), 'observation_q':np.asarray(formatted_observation['observation.state']).tolist(),
                             'predicted_actions':np.asarray(action).tolist(),
                             'execute_length':int(os.environ.get('ROUND1_EXECUTE_LENGTH','50'))})
            if len(action.shape) == 2:
                initial_obs = False
                for action_index, act in enumerate(action[:int(os.environ.get('ROUND1_EXECUTE_LENGTH','50'))]):
                    if initial_obs: # ensure the video is correct, but slow down simulation
                        if TASK_ENV.eval_video_path is not None:
                            heartbeat('chunk_get_obs')
                            TASK_ENV.get_obs()  # Refresh recorded frame within each action chunk.
                    else:
                        initial_obs = True
                    heartbeat('take_action')
                    diagnostic_execute(TASK_ENV, act, diag_path, chunk_id, action_index)
                    if TASK_ENV.eval_success:
                        succ = True
                        break
            else:
                heartbeat('take_action')
                diagnostic_execute(TASK_ENV, action, diag_path, chunk_id, 0)
                if TASK_ENV.eval_success:
                    succ = True
            
            chunk_id += 1
            print(f"infer time {latency}")


        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            heartbeat('video_finalize',True)
            TASK_ENV._del_eval_video_ffmpeg()
            if ffmpeg.returncode != 0:
                raise RuntimeError(f"Video encoder failed: {ffmpeg.returncode}")

        result_tag = "success" if succ else "failure"
        
        old_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4"
        new_name = f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}_{result_tag}.mp4"

        if os.path.exists(old_name):
            os.rename(old_name, new_name)
            print(f"Video saved: {new_name}")


        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        diagnostic_write(diag_path, {'kind':'episode_end', 'success':bool(succ), 'steps':int(TASK_ENV.take_action_cnt)})
        record = {'diagnostic':str(diag_path), 'episode': TASK_ENV.test_num, 'seed': now_seed, 'success': bool(succ),
                  'steps': int(TASK_ENV.take_action_cnt), 'step_limit': int(TASK_ENV.step_lim),
                  'instruction': instruction, 'seconds': time.monotonic()-episode_started,
                  'video': new_name if os.path.exists(new_name) else None}
        with open(Path(args['baseline_save_dir'])/'episodes.jsonl', 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
            f.flush()
            os.fsync(f.fileno())
        heartbeat('episode_saved',True)
        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    # from test_render import Sapien_TEST
    # Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
