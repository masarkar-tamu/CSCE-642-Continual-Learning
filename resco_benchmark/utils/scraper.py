import os

from bs4 import BeautifulSoup
import datetime
import time as thred

import requests


for signal_id in ["7142", "7243", "7241", "7242"]:  #
    start_date = datetime.datetime(2023, 1, 1)
    os.mkdir(os.curdir + os.sep + signal_id)
    while start_date < datetime.datetime(2024, 5, 3):
        check_date = start_date.strftime("%#m/%#d/%Y")
        x = requests.post(
            "https://x.udottraffic.utah.gov/ATSPM/DefaultCharts/GetTMCMetric",
            json={
                "SignalID": signal_id,
                "StartDate": check_date + " 12:00 AM",
                "EndDate": check_date + " 11:59 PM",
                "YAxisMax": "1000",
                "Y2AxisMax": "300",
                "MetricTypeID": 5,
                "SelectedBinSize": "5",
                "ShowLaneVolumes": "true",
                "ShowTotalVolumes": "false",
                "ShowDataTable": "true",
            },
        )

        col_indices = dict()
        directions = list()
        keep_looking = True
        cur_direction = None
        offset = 0

        soup = BeautifulSoup(x.text, "html.parser")
        demand_table = soup.table
        headers = demand_table.find_all("th")
        for i, header in enumerate(headers):
            header = header.contents[0].strip().upper()

            # Table has a blank over vehicle total
            if len(header) == 0:
                keep_looking = False
                saved_directions = directions.copy()
                for direction in directions:
                    col_indices[direction] = dict()
            if keep_looking and header not in [
                "EASTBOUND",
                "WESTBOUND",
                "NORTHBOUND",
                "SOUTHBOUND",
            ]:
                continue
            if keep_looking:
                directions.append(header)

            if not keep_looking:
                if cur_direction is None and header in [
                    "L",
                    "T",
                    "R",
                    "TR",
                ]:  # Found the first turning count
                    offset = i - 1
                    cur_direction = directions.pop(0)
                    col_indices[cur_direction][header] = 1
                if cur_direction is not None:
                    if "TOTAL" in header:
                        if len(directions) == 0:
                            break
                        cur_direction = directions.pop(0)
                    else:
                        col_indices[cur_direction][header] = i - offset

        rows = demand_table.find_all("tr")
        cleaned = list()
        for row in rows:
            row = row.find_all("td")
            csvd = list()
            for i, col in enumerate(row):
                for child in col.children:
                    try:
                        csvd.append(int(child))
                    except:
                        if "AM" in str(col) or "PM" in str(col):
                            csvd.append(" ".join(str(child).replace("\n", "").split()))

            if "AM" in str(csvd) or "PM" in str(csvd):
                cleaned.append(csvd)
        print(len(cleaned))

        def read_time(tm):
            return datetime.datetime.strptime(tm, "%m/%d/%Y %I:%M %p").strftime(
                "%Y-%m-%d %H:%M"
            )

        csv_lines = []
        prev_time = None
        a_line = []
        for line in cleaned:
            for direction in saved_directions:
                for movement in ["L", "T", "R", "TR"]:
                    if movement in col_indices[direction]:
                        time = line[0]
                        date = start_date.strftime("%#m/%#d/%Y")
                        cur_time = read_time(date + " " + time)
                        demand = line[col_indices[direction][movement]]
                        if prev_time is not None:
                            if cur_time != prev_time:
                                print(str(prev_time) + "," + ",".join(a_line))
                                csv_lines.append(
                                    str(prev_time) + "," + ",".join(a_line)
                                )
                                a_line = []
                        a_line.append(str(demand))
                        prev_time = cur_time
        print(str(prev_time) + "," + ",".join(a_line))
        csv_lines.append(str(prev_time) + "," + ",".join(a_line))

        # write csv lines ot file
        with open(
            os.curdir
            + os.sep
            + signal_id
            + os.sep
            + start_date.strftime("%Y-%m-%d")
            + ".csv",
            "w",
        ) as file:
            for line in csv_lines:
                file.write(line + "\n")
        start_date += datetime.timedelta(days=1)
        thred.sleep(2)
