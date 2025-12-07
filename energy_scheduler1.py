HIGH_FREQ = 2.0
MID_FREQ = 1.7
LOW_FREQ = 1.3

THRESHOLD_HIGH = 10
THRESHOLD_MID = 5

"""
Energy-efficient CPU scheduling simulator.

Implements FCFS, Round Robin, and an energy-aware DVFS scheduler,
and compares them based on waiting time, turnaround time, utilization,
and energy consumption.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Store results of the last single run (for trend lines)
LAST_RESULTS: List[Dict[str, float]] = []


# ------------------------------
# Data Models
# ------------------------------

@dataclass
class Process:
    pid: str
    arrival: int
    burst: int
    remaining: int = field(init=False)
    start_time: int = None
    completion_time: int = None
    waiting_time: int = None
    turnaround_time: int = None

    def __post_init__(self):
        self.remaining = self.burst


@dataclass
class ExecSegment:
    pid: str
    start: int
    end: int
    freq: float  # CPU frequency used during this segment


# ------------------------------
# Scheduler Base Class
# ------------------------------

class SchedulerBase:
    name: str = "BaseScheduler"

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        raise NotImplementedError("Subclasses must implement run()")


# ------------------------------
# FCFS Scheduler (Baseline)
# ------------------------------

class FCFSScheduler(SchedulerBase):
"""Non-preemptive First Come First Serve scheduler."""
    name = "FCFS (High Frequency)"

    def __init__(self, freq: float = 2.0):
        self.freq = freq

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        procs = sorted(
            [Process(p.pid, p.arrival, p.burst) for p in processes],
            key=lambda p: (p.arrival, p.pid),
        )
        time = 0
        segments: List[ExecSegment] = []

        for p in procs:
            if time < p.arrival:
                time = p.arrival
            p.start_time = time
            start = time

            exec_time = p.burst
            time += exec_time

            p.completion_time = time
            p.turnaround_time = p.completion_time - p.arrival
            p.waiting_time = p.turnaround_time - p.burst

            segments.append(ExecSegment(p.pid, start, time, self.freq))

        return procs, segments


# ------------------------------
# Round Robin Scheduler
# ------------------------------

class RRScheduler(SchedulerBase):
    name = "Round Robin (High Frequency)"

    def __init__(self, quantum: int = 2, freq: float = 2.0):
        self.quantum = quantum
        self.freq = freq

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        procs = [Process(p.pid, p.arrival, p.burst) for p in processes]
        time = 0
        segments: List[ExecSegment] = []
        ready: List[Process] = []
        procs_sorted = sorted(procs, key=lambda p: (p.arrival, p.pid))

        i = 0
        completed = 0
        n = len(procs_sorted)

        while completed < n:
            while i < n and procs_sorted[i].arrival <= time:
                ready.append(procs_sorted[i])
                i += 1

            if not ready:
                time = procs_sorted[i].arrival
                continue

            current = ready.pop(0)

            if current.start_time is None:
                current.start_time = time

            run_time = min(self.quantum, current.remaining)
            start = time
            time += run_time
            current.remaining -= run_time

            segments.append(ExecSegment(current.pid, start, time, self.freq))

            while i < n and procs_sorted[i].arrival <= time:
                ready.append(procs_sorted[i])
                i += 1

            if current.remaining > 0:
                ready.append(current)
            else:
                current.completion_time = time
                current.turnaround_time = current.completion_time - current.arrival
                current.waiting_time = current.turnaround_time - current.burst
                completed += 1

        return procs_sorted, segments


# ------------------------------
# Energy-Aware Scheduler (DVFS)
# ------------------------------

class EnergyAwareScheduler(SchedulerBase):
    name = "Energy-Aware FCFS with DVFS"

    def __init__(
        self,
        high_freq: float = 2.0,
        mid_freq: float = 1.7,
        low_freq: float = 1.3,
        threshold_high: int = 10,
        threshold_mid: int = 5,
    ):
        self.high_freq = high_freq
        self.mid_freq = mid_freq
        self.low_freq = low_freq
        self.threshold_high = threshold_high
        self.threshold_mid = threshold_mid

    def run(self, processes: List[Process]) -> Tuple[List[Process], List[ExecSegment]]:
        procs = sorted(
            [Process(p.pid, p.arrival, p.burst) for p in processes],
            key=lambda p: (p.arrival, p.pid),
        )
        time = 0
        segments: List[ExecSegment] = []
        base_freq = self.high_freq

        for idx, p in enumerate(procs):
            if time < p.arrival:
                time = p.arrival

            total_remaining = sum(x.burst for x in procs[idx:])

            if total_remaining > self.threshold_high:
                freq = self.high_freq
            elif total_remaining > self.threshold_mid:
                freq = self.mid_freq
            else:
                freq = self.low_freq

            effective_speed = freq / base_freq
            exec_time = math.ceil(p.burst / effective_speed)

            p.start_time = time
            start = time
            time += exec_time
            p.completion_time = time
            p.turnaround_time = p.completion_time - p.arrival
            p.waiting_time = p.turnaround_time - p.burst

            segments.append(ExecSegment(p.pid, start, time, freq))

        return procs, segments


# ------------------------------
# Energy Model & Metrics
# ------------------------------

def power(freq: float) -> float:
    return freq ** 3


def compute_metrics(procs: List[Process], segments: List[ExecSegment]) -> Dict[str, float]:
    n = len(procs)
    avg_wait = sum(p.waiting_time for p in procs) / n
    avg_turn = sum(p.turnaround_time for p in procs) / n

    makespan = max(p.completion_time for p in procs) - min(p.arrival for p in procs)
    busy = sum(seg.end - seg.start for seg in segments)
    util = busy / makespan if makespan > 0 else 0.0

    energy = sum(power(seg.freq) * (seg.end - seg.start) for seg in segments)

    return {
        "avg_waiting_time": avg_wait,
        "avg_turnaround_time": avg_turn,
        "cpu_utilization": util,
        "total_energy": energy,
    }


# ------------------------------
# Pretty Printing Helpers
# ------------------------------

def print_process_table(processes: List[Process]) -> None:
    print("\nProcesses:")
    print("+-------+----------+--------+")
    print("| PID   | Arrival  | Burst  |")
    print("+-------+----------+--------+")
    for p in processes:
        print(f"| {p.pid:<5} | {p.arrival:^8} | {p.burst:^6} |")
    print("+-------+----------+--------+\n")


def print_metrics_table(metrics: Dict[str, float]) -> None:
    print("Metrics:")
    print("+----------------------+------------+")
    print(f"| {'Avg waiting time':<20} | {metrics['avg_waiting_time']:>8.2f} |")
    print(f"| {'Avg turnaround':<20}  | {metrics['avg_turnaround_time']:>8.2f} |")
    print(f"| {'CPU utilization':<20} | {metrics['cpu_utilization']*100:>7.2f}% |")
    print(f"| {'Total energy':<20}    | {metrics['total_energy']:>8.2f} |")
    print("+----------------------+------------+\n")


def print_gantt(segments: List[ExecSegment], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not segments:
        print("(no segments)\n")
        return

    print("Each character ≈ 1 time unit.")
    print("Top line: which process ran.")
    print("Second line: frequency level (H=High, M=Medium, L=Low).\n")

    timeline = ""
    freq_line = ""
    time_marks = ""
    last_end = segments[0].start

    for seg in segments:
        if seg.start > last_end:
            gap = seg.start - last_end
            timeline += " " * gap
            freq_line += " " * gap
            for t in range(last_end, seg.start):
                time_marks += "|" if t % 5 == 0 else " "

        run_len = seg.end - seg.start
        block = seg.pid * run_len
        timeline += block

        if seg.freq >= 1.9:
            freq_char = "H"
        elif seg.freq >= 1.5:
            freq_char = "M"
        else:
            freq_char = "L"
        freq_line += freq_char * run_len

        for t in range(seg.start, seg.end):
            time_marks += "|" if t % 5 == 0 else " "

        last_end = seg.end

    print("Processes:")
    print(timeline)
    print("Freq lvl:")
    print(freq_line)
    print(time_marks)
    print("Legend: H=High freq, M=Medium, L=Low, '|' every 5 time units\n")


# ------------------------------
# Plot Helpers
# ------------------------------

def show_energy_bar_chart(results: List[Dict[str, float]]) -> None:
    names = [r["name"] for r in results]
    energies = [r["energy"] for r in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, energies)

    plt.title("Energy Consumption Comparison")
    plt.xlabel("Scheduling Algorithm")
    plt.ylabel("Total Energy (units)")

    for bar, energy in zip(bars, energies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{energy:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def show_energy_trend_line() -> None:
    if not LAST_RESULTS:
        print("No results available yet. Run an experiment first (option 1 or 2).\n")
        return

    names = [r["name"] for r in LAST_RESULTS]
    energies = [r["energy"] for r in LAST_RESULTS]

    plt.figure(figsize=(6, 4))
    plt.plot(names, energies, marker="o")
    plt.title("Trend: Energy vs Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Total Energy (units)")
    plt.tight_layout()
    plt.show()


def show_tat_trend_line() -> None:
    if not LAST_RESULTS:
        print("No results available yet. Run an experiment first (option 1 or 2).\n")
        return

    names = [r["name"] for r in LAST_RESULTS]
    tats = [r["avg_tat"] for r in LAST_RESULTS]

    plt.figure(figsize=(6, 4))
    plt.plot(names, tats, marker="o")
    plt.title("Trend: Avg Turnaround Time vs Algorithm")
    plt.xlabel("Algorithm")
    plt.ylabel("Avg Turnaround Time (units)")
    plt.tight_layout()
    plt.show()


def show_dvfs_logic_diagram() -> None:
    """
    Show a simple block diagram of the DVFS decision logic.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axis_off()

    y_positions = [2.2, 1.2, 0.2]
    loads = [
        "High Load\n(total remaining > 10)",
        "Medium Load\n(5 < total remaining ≤ 10)",
        "Low Load\n(total remaining ≤ 5)",
    ]
    freqs = [
        "High Frequency\n2.0 GHz",
        "Medium Frequency\n1.7 GHz",
        "Low Frequency\n1.3 GHz",
    ]

    for y, load_text, freq_text in zip(y_positions, loads, freqs):
        load_box = Rectangle((0.1, y), 0.4, 0.6, edgecolor="black", facecolor="#e0f0ff")
        freq_box = Rectangle((0.6, y), 0.4, 0.6, edgecolor="black", facecolor="#e0ffe0")
        ax.add_patch(load_box)
        ax.add_patch(freq_box)

        ax.text(0.3, y + 0.3, load_text, ha="center", va="center", fontsize=9)
        ax.text(0.8, y + 0.3, freq_text, ha="center", va="center", fontsize=9)

        arrow = FancyArrowPatch(
            (0.5, y + 0.3),
            (0.6, y + 0.3),
            arrowstyle="->",
            mutation_scale=15,
        )
        ax.add_patch(arrow)

    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 3.2)
    plt.title("DVFS Logic: Mapping Load to CPU Frequency")
    plt.tight_layout()
    plt.show()


def show_flow_diagram() -> None:
    """
    Show a simple flow diagram of the simulator steps.
    """
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.set_axis_off()

    steps = [
        "Start",
        "Get Processes\n(sample or custom)",
        "Run FCFS, RR,\nDVFS Scheduler",
        "Compute Metrics\n(wait, TAT, energy)",
        "Display Table\n+ Graphs",
        "End",
    ]

    y_positions = list(reversed(range(len(steps))))

    for i, (step, y) in enumerate(zip(steps, y_positions)):
        box = Rectangle((0.2, y), 0.6, 0.8, edgecolor="black", facecolor="#fff0d9")
        ax.add_patch(box)
        ax.text(0.5, y + 0.4, step, ha="center", va="center", fontsize=9)

        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (0.5, y),
                (0.5, y - 0.2),
                arrowstyle="->",
                mutation_scale=15,
            )
            ax.add_patch(arrow)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(steps) + 0.5)
    plt.title("Flow of the Energy-Efficient CPU Scheduling Simulator")
    plt.tight_layout()
    plt.show()


def show_combined_gantt(segments_by_algo: Dict[str, List[ExecSegment]]) -> None:
    """
    Draw a combined colourful Gantt chart for all algorithms in one figure.
    Each algorithm gets its own horizontal row.
    """
    if not segments_by_algo:
        print("No segments available for Gantt chart.\n")
        return

    # Collect all PIDs to assign consistent colours
    all_pids = sorted(
        {seg.pid for segs in segments_by_algo.values() for seg in segs}
    )
    color_cycle = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
    ]
    pid_colors = {
        pid: color_cycle[i % len(color_cycle)] for i, pid in enumerate(all_pids)
    }

    fig, ax = plt.subplots(figsize=(9, 4))

    algo_names = list(segments_by_algo.keys())
    y_positions = list(range(len(algo_names)))  # 0,1,2,...

    # Draw bars for each algorithm row
    for y, algo in zip(y_positions, algo_names):
        segs = segments_by_algo[algo]
        for seg in segs:
            start = seg.start
            duration = seg.end - seg.start
            color = pid_colors.get(seg.pid, "tab:gray")
            ax.barh(
                y,
                duration,
                left=start,
                height=0.4,
                edgecolor="black",
                align="center",
                color=color,
            )
            # Put PID label in the middle of the bar
            ax.text(
                start + duration / 2,
                y,
                seg.pid,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if color != "tab:orange" else "black",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(algo_names)
    ax.set_xlabel("Time")
    ax.set_title("Combined Gantt Chart – FCFS vs RR vs DVFS")

    # Build a small PID legend
    handles = []
    from matplotlib.patches import Patch
    for pid in all_pids:
        handles.append(Patch(facecolor=pid_colors[pid], edgecolor="black", label=f"PID {pid}"))
    ax.legend(handles=handles, title="Processes", bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


# ------------------------------
# Input Helper
# ------------------------------

def read_processes_from_user() -> List[Process]:
    print("\nEnter process details.")
    while True:
        try:
            n = int(input("Number of processes: ").strip())
            if n <= 0:
                print("Please enter a positive number.\n")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.\n")

    processes: List[Process] = []
    print("Enter each process as: <PID> <arrival_time> <burst_time>")
    print("Example: A 0 4")

    for i in range(n):
        while True:
            line = input(f"Process {i+1}: ").strip()
            parts = line.split()
            if len(parts) != 3:
                print("Invalid format. Use: PID arrival burst (3 values). Try again.")
                continue
            pid = parts[0]
            try:
                arrival = int(parts[1])
                burst = int(parts[2])
                if burst <= 0:
                    print("Burst time must be > 0. Try again.")
                    continue
            except ValueError:
                print("Arrival and burst must be integers. Try again.")
                continue
            processes.append(Process(pid, arrival, burst))
            break

    return processes


# ------------------------------
# Core Result Computation
# ------------------------------

def compute_results_for_processes(processes: List[Process]) -> List[Dict[str, float]]:
    """
    Run all schedulers on the given processes and return a list of
    dictionaries(summary list) with metrics for each algorithm.
    """
    schedulers: List[SchedulerBase] = [
        FCFSScheduler(freq=2.0),
        RRScheduler(quantum=2, freq=2.0),
        EnergyAwareScheduler(
            high_freq=2.0,
            mid_freq=1.7,
            low_freq=1.3,
            threshold_high=10,
            threshold_mid=5,
        ),
    ]

    base_metrics = None
    results: List[Dict[str, float]] = []

    for idx, sched in enumerate(schedulers):
        procs, segs = sched.run(processes)
        metrics = compute_metrics(procs, segs)

        if idx == 0:
            base_metrics = metrics
            d_energy = 0.0
            d_tat = 0.0
        else:
            d_energy = (
                (metrics["total_energy"] - base_metrics["total_energy"])
                / base_metrics["total_energy"]
                * 100
            )
            d_tat = (
                (metrics["avg_turnaround_time"] - base_metrics["avg_turnaround_time"])
                / base_metrics["avg_turnaround_time"]
                * 100
            )

        results.append(
            {
                "name": sched.name,
                "avg_wait": metrics["avg_waiting_time"],
                "avg_tat": metrics["avg_turnaround_time"],
                "util": metrics["cpu_utilization"] * 100,
                "energy": metrics["total_energy"],
                "d_energy": d_energy,
                "d_tat": d_tat,
            }
        )

    return results


# ------------------------------
# Experiment Runner (single run)
# ------------------------------

def run_experiment(processes: List[Process]) -> None:
    global LAST_RESULTS

    # 1) Show input processes (for screenshot + confirmation)
    print_process_table(processes)

    # Confirmation before running
    while True:
        ans = input("Proceed with these processes? (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            break
        elif ans in ("n", "no"):
            print("Cancelled. Returning to Menu.\n")
            return
        else:
            print("Please enter 'y' or 'n'.\n")

    # 2) Compute results for all algorithms (metrics only)
    results = compute_results_for_processes(processes)
    LAST_RESULTS = results  # save for trend-line graphs

    # 3) Compact comparison table
    print("=== Algorithm Comparison Summary ===")
    print("+-------------------------------+----------+----------+----------+-----------+------------+------------+")
    print("| Algorithm                     | AvgWait  | AvgTAT   | Util(%)  | Energy    | ΔEnergy %  | ΔTAT %    |")
    print("+-------------------------------+----------+----------+----------+-----------+------------+------------+")
    for r in results:
        print(
            f"| {r['name']:<29}"
            f"| {r['avg_wait']:>8.2f} "
            f"| {r['avg_tat']:>8.2f} "
            f"| {r['util']:>8.2f} "
            f"| {r['energy']:>9.2f} "
            f"| {r['d_energy']:>10.2f} "
            f"| {r['d_tat']:>10.2f} |"
        )
    print("+-------------------------------+----------+----------+----------+-----------+------------+------------+\n")

    # 4) Quick summary text
    print("Quick Summary:")
    print(f"- You tested {len(processes)} processes.")
    algo_names = ", ".join(r["name"] for r in results)
    print(f"- Algorithms compared: {algo_names}")

    best_energy = min(results, key=lambda r: r["energy"])
    best_tat = min(results, key=lambda r: r["avg_tat"])

    print(f"- Lowest energy: {best_energy['name']} ({best_energy['energy']:.2f} units)")
    print(f"- Best (lowest) average turnaround time: {best_tat['name']} ({best_tat['avg_tat']:.2f} units)")
    print("- This is the output of energy-scheduling algorithm.\n")

    # 5) Show bar chart for energy
    try:
        show_energy_bar_chart(results)
    except Exception as e:
        print("Could not display energy bar chart:", e)

    # 6) Build a combined colourful Gantt chart for all algorithms
    try:
        # Re-run each scheduler to collect segments (small overhead, but OK for a project)
        schedulers: List[SchedulerBase] = [
            FCFSScheduler(freq=2.0),
            RRScheduler(quantum=2, freq=2.0),
            EnergyAwareScheduler(
    		high_freq=HIGH_FREQ,
    		mid_freq=MID_FREQ,
    		low_freq=LOW_FREQ,
    		threshold_high=THRESHOLD_HIGH,
    		threshold_mid=THRESHOLD_MID,
		),
        ]

        segments_by_algo: Dict[str, List[ExecSegment]] = {}
        for sched in schedulers:
            # Important: give a *fresh copy* of processes each time
            proc_copy = [Process(p.pid, p.arrival, p.burst) for p in processes]
            _, segs = sched.run(proc_copy)
            segments_by_algo[sched.name] = segs

        show_combined_gantt(segments_by_algo)
    except Exception as e:
        print("Could not display combined Gantt chart:", e)


# ------------------------------
# Multi-run Trend Graphs
# ------------------------------

def run_multi_trend_sample() -> None:
    """
    Run the schedulers on several predefined workloads (small/medium/heavy)
    and show trend lines for energy and average turnaround time.
    """
    print("\nRunning multi-run trend graphs on SAMPLE workloads...")

    sample_workloads = [
        (
            "Small",
            [
                Process("A", 0, 4),
                Process("B", 1, 5),
                Process("C", 2, 3),
                Process("D", 4, 2),
            ],
        ),
        (
            "Medium",
            [
                Process("A", 0, 3),
                Process("B", 2, 6),
                Process("C", 4, 4),
                Process("D", 6, 5),
                Process("E", 8, 2),
            ],
        ),
        (
            "Heavy",
            [
                Process("A", 0, 5),
                Process("B", 1, 7),
                Process("C", 3, 6),
                Process("D", 5, 4),
                Process("E", 6, 3),
                Process("F", 8, 2),
            ],
        ),
    ]

    labels = [name for name, _ in sample_workloads]
    all_results: List[List[Dict[str, float]]] = []

    for label, procs in sample_workloads:
        results = compute_results_for_processes(procs)
        all_results.append(results)

    if not all_results:
        print("No workloads defined.\n")
        return

    algo_names = [r["name"] for r in all_results[0]]
    energy_trends = {name: [] for name in algo_names}
    tat_trends = {name: [] for name in algo_names}

    for results in all_results:
        for r in results:
            energy_trends[r["name"]].append(r["energy"])
            tat_trends[r["name"]].append(r["avg_tat"])

    try:
        # Energy vs workload size
        plt.figure(figsize=(6, 4))
        for name in algo_names:
            plt.plot(labels, energy_trends[name], marker="o", label=name)
        plt.title("Energy vs Workload Size (Sample Sets)")
        plt.xlabel("Workload")
        plt.ylabel("Total Energy (units)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Turnaround vs workload size
        plt.figure(figsize=(6, 4))
        for name in algo_names:
            plt.plot(labels, tat_trends[name], marker="o", label=name)
        plt.title("Average Turnaround Time vs Workload Size (Sample Sets)")
        plt.xlabel("Workload")
        plt.ylabel("Avg Turnaround Time (units)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not display multi-run trend graphs:", e)

    print("Finished multi-run trend graphs for sample workloads.\n")


def run_multi_trend_custom() -> None:
    """
    Ask the user for multiple custom workloads and plot trend lines
    for energy and average turnaround time across those workloads.
    """
    print("\nMulti-run trend graphs (CUSTOM workloads)")
    while True:
        try:
            k = int(input("How many different workload sets do you want to compare? ").strip())
            if k <= 0:
                print("Please enter a positive number.\n")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.\n")

    workload_labels: List[str] = []
    all_results: List[List[Dict[str, float]]] = []

    for i in range(k):
        label = input(f"\nLabel for workload set {i+1} (e.g. 'Case 1', 'Small'): ").strip()
        if not label:
            label = f"Set {i+1}"
        print(f"Enter processes for {label}:")
        processes = read_processes_from_user()
        workload_labels.append(label)
        results = compute_results_for_processes(processes)
        all_results.append(results)

    if not all_results:
        print("No workloads entered.\n")
        return

    algo_names = [r["name"] for r in all_results[0]]
    energy_trends = {name: [] for name in algo_names}
    tat_trends = {name: [] for name in algo_names}

    for results in all_results:
        for r in results:
            energy_trends[r["name"]].append(r["energy"])
            tat_trends[r["name"]].append(r["avg_tat"])

    try:
        # Energy vs workload label
        plt.figure(figsize=(6, 4))
        for name in algo_names:
            plt.plot(workload_labels, energy_trends[name], marker="o", label=name)
        plt.title("Energy vs Workload (Custom Sets)")
        plt.xlabel("Workload Set")
        plt.ylabel("Total Energy (units)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Turnaround vs workload label
        plt.figure(figsize=(6, 4))
        for name in algo_names:
            plt.plot(workload_labels, tat_trends[name], marker="o", label=name)
        plt.title("Average Turnaround Time vs Workload (Custom Sets)")
        plt.xlabel("Workload Set")
        plt.ylabel("Avg Turnaround Time (units)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Could not display custom multi-run trend graphs:", e)

    print("Finished multi-run trend graphs for custom workloads.\n")


# ------------------------------
# Main Menu
# ------------------------------

def main() -> None:
    print("=" * 60)
    print("        ENERGY EFFICIENT CPU SCHEDULING SIMULATOR")
    print("=" * 60)
    print("This tool compares classic CPU schedulers (FCFS, Round Robin)")
    print("with an Energy-Aware DVFS-based scheduler.\n")

    while True:
        print("Menu:")
        print("  1. Use sample processes")
        print("  2. Enter custom processes")
        print("  3. Multi-run trend graphs (sample workloads)")
        print("  4. Multi-run trend graphs (custom workloads)")
        print("  5. Show DVFS logic diagram")
        print("  6. Show simulator flow diagram")
        print("  7. Energy trend line (last run)")
        print("  8. Turnaround time trend line (last run)")
        print("  9. Exit")
        choice = input("Choose an option (1–9): ").strip()

        if choice == "1":
            processes = [
                Process("A", 0, 4),
                Process("B", 1, 5),
                Process("C", 2, 3),
                Process("D", 4, 2),
            ]
            print("\nUsing sample processes.")
            run_experiment(processes)

        elif choice == "2":
            processes = read_processes_from_user()
            run_experiment(processes)

        elif choice == "3":
            run_multi_trend_sample()

        elif choice == "4":
            run_multi_trend_custom()

        elif choice == "5":
            show_dvfs_logic_diagram()

        elif choice == "6":
            show_flow_diagram()

        elif choice == "7":
            show_energy_trend_line()

        elif choice == "8":
            show_tat_trend_line()

        elif choice == "9":
            print("Exiting simulator. Goodbye!")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 9.\n")


if __name__ == "__main__":
    main()
