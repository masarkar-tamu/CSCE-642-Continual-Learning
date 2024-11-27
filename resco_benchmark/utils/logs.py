import os
import shutil
import logging
import json
import subprocess
import xml.etree.ElementTree as ET

from resco_benchmark.config.config import config as cfg, hash_config

logger = logging.getLogger(__name__)


# target_dir bypasses default of running UUID
# no_compress forces compression skip regardless of config parameters
def parse_logs(target_dir=None, no_compress=False):
    if target_dir is not None:
        # When called from util scripts config can't be set for each run
        uuid = target_dir[target_dir.rfind(os.sep) :].replace(os.sep, "")
        # Run name is two directories up from the UUID
        run_name = (
            target_dir[: target_dir[: target_dir.rfind(os.sep)].rfind(os.sep)]
            + uuid[-8:]
        )
        # Hash the config to get a unique name
        run_path = target_dir
        config_fp = os.path.join(target_dir, "config.json")
        with open(config_fp) as f:
            json_in = json.load(f)
        hashed_name = hash_config(json_in)
    else:
        run_path = cfg.run_path
        hashed_name = cfg.hashed_name
        run_name = cfg.run_name + cfg.uuid[-8:]

    csv_met_map, eps_avgs, results = dict(), dict(), dict()
    for i, met in enumerate(cfg.csv_metrics):
        csv_met_map[met] = i
    for metric in cfg.xml_metrics + cfg.csv_metrics:
        eps_avgs[metric] = list()
        results[metric] = dict()

    i = 1
    eps_avgs["RETURNFLAG"] = True
    while eps_avgs["RETURNFLAG"]:
        parse_xml_log(run_path, i, eps_avgs)
        parse_csv_log(run_path, i, eps_avgs, csv_met_map)
        i += 1

    # Create new JSON log per run_name
    for metric in cfg.xml_metrics + cfg.csv_metrics:
        results[metric][hashed_name] = [eps_avgs[metric]]

    print("Saving results to:", os.path.join(cfg.log_dir, run_name + ".json"))
    with open(os.path.join(cfg.log_dir, run_name + ".json"), "w") as f:
        json.dump(results, f)

    if cfg.compress_results and not no_compress:
        compress_folder(run_path)
    return eps_avgs


def parse_xml_log(target_dir, i, eps_avgs):
    # Loop until next file (i+1) does not exist
    trip_file_name = os.path.join(target_dir, "tripinfo_{0}.xml".format(i))
    if not os.path.exists(trip_file_name):
        eps_avgs["RETURNFLAG"] = False
        return False

    # Deformed XML is sometimes output, don't let it stop the rest of the process
    # Happens when running on active processes
    try:
        tree = ET.parse(trip_file_name)
    except ET.ParseError:
        eps_avgs["RETURNFLAG"] = False
        return False

    # Read SUMO output XMLs
    root = tree.getroot()
    num_trips = 0
    totals = dict()
    for metric in cfg.xml_metrics:
        totals[metric] = 0.0
    for child in root:
        if child.attrib["id"].startswith("ghost"):
            continue
        num_trips += 1
        for metric in cfg.xml_metrics:
            totals[metric] += float(child.attrib[metric])
            if metric == "timeLoss":
                totals[metric] += float(child.attrib["departDelay"])
    if num_trips == 0:
        num_trips = 1

    for metric in cfg.xml_metrics:
        eps_avgs[metric].append(totals[metric] / num_trips)
    return True


# RESCO CSV handling
def parse_csv_log(target_dir, i, eps_avgs, csv_met_map):
    trip_file_name = os.path.join(target_dir, "metrics_{0}.csv".format(i))
    if not os.path.exists(trip_file_name):
        logger.info("{0} has {1} episodes".format(target_dir, i - 1))
        eps_avgs["RETURNFLAG"] = False
        return False
    with open(trip_file_name) as fp:
        num_steps = 0
        totals = dict()
        for metric in cfg.csv_metrics:
            totals[metric] = 0.0
        next(fp)  # Skip header
        for line in fp:
            line = line.split("}")
            num_steps += 1
            for metric in cfg.csv_metrics:
                queues = line[csv_met_map[metric]]
                signals = queues.split(":")
                step_total = 0
                for s, signal in enumerate(signals):
                    if s == 0:
                        continue
                    queue = signal.split(",")
                    queue = float(queue[0])
                    step_total += queue
                step_avg = step_total / s
                totals[metric] += step_avg
        # if num_steps == 0: num_steps = 1   TODO why is this here?
        for metric in cfg.csv_metrics:
            eps_avgs[metric].append(totals[metric] / num_steps)
    return True


def compress_folder(folder):
    logger.info("Compressing {0}".format(folder))
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()  # Release console.log for deletion

    folder = folder[: folder.rfind(os.sep)]  # Remove trailing slash

    if cfg.delete_logs:
        shutil.rmtree(folder, ignore_errors=True)
    elif shutil.which("pigz") is not None:
        cmd = 'nohup tar --remove-files -I pigz -cf "{0}.tar.gz" "{1}" &'.format(
            folder, folder
        )
        subprocess.call(cmd, shell=True)
    else:
        shutil.make_archive(folder, "zip", folder)

        shutil.rmtree(folder)
