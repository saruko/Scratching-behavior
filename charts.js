/**
 * Mouse Scratch Detection System - Dashboard Charts
 * =================================================
 * Orchestrates Chart.js instances for:
 *   1. Real-time Telemetry (rolling kinematic curves of selected mouse)
 *   2. Analytics (comparative scratch bouts count & duration summaries)
 */

class DashboardCharts {
    constructor() {
        this.signalChart = null;
        this.boutCountChart = null;
        this.boutDurationChart = null;
        
        this.mouseColors = {
            1: "rgba(239, 68, 68, 0.85)",   // Red
            2: "rgba(59, 130, 246, 0.85)",  // Blue
            3: "rgba(234, 179, 8, 0.85)",   // Yellow
            4: "rgba(156, 163, 175, 0.85)"  // Gray
        };
    }

    /**
     * Initialize the rolling real-time signal chart in the sidebar
     */
    initSignalChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.signalChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({ length: 30 }, (_, i) => -29 + i), // -29 to 0 frames
                datasets: [
                    {
                        label: '左後肢 速度 (px/f)',
                        data: Array(30).fill(0),
                        borderColor: '#00F2FE',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0,
                        yAxisID: 'ySpeed'
                    },
                    {
                        label: '右後肢 速度 (px/f)',
                        data: Array(30).fill(0),
                        borderColor: '#A855F7',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0,
                        yAxisID: 'ySpeed'
                    },
                    {
                        label: '左耳-左後肢 距離 (px)',
                        data: Array(30).fill(150),
                        borderColor: 'rgba(239, 68, 68, 0.8)',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        tension: 0.3,
                        pointRadius: 0,
                        yAxisID: 'yDist'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#9CA3AF',
                            font: { size: 10, family: 'Inter' },
                            boxWidth: 10,
                            padding: 8
                        },
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#6B7280', font: { size: 9 } }
                    },
                    ySpeed: {
                        type: 'linear',
                        position: 'left',
                        min: 0,
                        max: 80,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9CA3AF', font: { size: 9 } },
                        title: { display: true, text: '速度', color: '#9CA3AF', font: { size: 9 } }
                    },
                    yDist: {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 200,
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#9CA3AF', font: { size: 9 } },
                        title: { display: true, text: '距離', color: '#9CA3AF', font: { size: 9 } }
                    }
                }
            }
        });
    }

    /**
     * Update the rolling signal chart with the latest 30 frames of data
     */
    updateSignalChart(rollingFrames, mouseId) {
        if (!this.signalChart || !rollingFrames || rollingFrames.length === 0) return;

        const speedL = [];
        const speedR = [];
        const distL = [];

        rollingFrames.forEach(f => {
            const mData = f.mice[mouseId];
            if (mData) {
                speedL.push(mData.features.hind_paw_speed_L || 0);
                speedR.push(mData.features.hind_paw_speed_R || 0);
                distL.push(mData.features.hind_to_ear_dist_L || 150);
            } else {
                speedL.push(0);
                speedR.push(0);
                distL.push(150);
            }
        });

        // Pad if less than 30 frames
        while (speedL.length < 30) {
            speedL.unshift(0);
            speedR.unshift(0);
            distL.unshift(150);
        }

        // Keep at most 30
        if (speedL.length > 30) {
            speedL.splice(0, speedL.length - 30);
            speedR.splice(0, speedR.length - 30);
            distL.splice(0, distL.length - 30);
        }

        this.signalChart.data.datasets[0].data = speedL;
        this.signalChart.data.datasets[1].data = speedR;
        this.signalChart.data.datasets[2].data = distL;
        
        // Dynamically adjust label for distance based on selected mouse ear
        this.signalChart.update('none'); // Update without animation for raw speed
    }

    /**
     * Generate or refresh the aggregated comparative analysis charts
     */
    updateAnalytics(bouts) {
        const mouseBoutCounts = { 1: 0, 2: 0, 3: 0, 4: 0 };
        const mouseBoutDurations = { 1: 0, 2: 0, 3: 0, 4: 0 };

        bouts.forEach(b => {
            const mId = b.mouse_id;
            if (mouseBoutCounts[mId] !== undefined) {
                mouseBoutCounts[mId]++;
                mouseBoutDurations[mId] += b.bout_duration_s;
            }
        });

        // Initialize or update Bout Count Bar Chart
        const countCtx = document.getElementById('chart-bout-counts');
        if (countCtx) {
            const dataCounts = [mouseBoutCounts[1], mouseBoutCounts[2], mouseBoutCounts[3], mouseBoutCounts[4]];
            
            if (this.boutCountChart) {
                this.boutCountChart.data.datasets[0].data = dataCounts;
                this.boutCountChart.update();
            } else {
                this.boutCountChart = new Chart(countCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Mouse 1 (赤)', 'Mouse 2 (青)', 'Mouse 3 (黄)', 'Mouse 4 (灰)'],
                        datasets: [{
                            label: 'ひっかき発生回数 (Bouts)',
                            data: dataCounts,
                            backgroundColor: [
                                this.mouseColors[1],
                                this.mouseColors[2],
                                this.mouseColors[3],
                                this.mouseColors[4]
                            ],
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                                ticks: { color: '#9CA3AF', stepSize: 1 }
                            },
                            x: {
                                ticks: { color: '#9CA3AF' },
                                grid: { display: false }
                            }
                        }
                    }
                });
            }
        }

        // Initialize or update Bout Cumulative Duration Doughnut Chart
        const durationCtx = document.getElementById('chart-bout-durations');
        if (durationCtx) {
            const dataDurations = [
                parseFloat(mouseBoutDurations[1].toFixed(2)),
                parseFloat(mouseBoutDurations[2].toFixed(2)),
                parseFloat(mouseBoutDurations[3].toFixed(2)),
                parseFloat(mouseBoutDurations[4].toFixed(2))
            ];

            if (this.boutDurationChart) {
                this.boutDurationChart.data.datasets[0].data = dataDurations;
                this.boutDurationChart.update();
            } else {
                this.boutDurationChart = new Chart(durationCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Mouse 1 (赤)', 'Mouse 2 (青)', 'Mouse 3 (黄)', 'Mouse 4 (灰)'],
                        datasets: [{
                            data: dataDurations,
                            backgroundColor: [
                                this.mouseColors[1],
                                this.mouseColors[2],
                                this.mouseColors[3],
                                this.mouseColors[4]
                            ],
                            borderColor: '#0B0F19',
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'right',
                                labels: {
                                    color: '#9CA3AF',
                                    font: { family: 'Inter', size: 11 },
                                    padding: 12
                                }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return ` ${context.label}: ${context.raw} 秒`;
                                    }
                                }
                            }
                        },
                        cutout: '60%'
                    }
                });
            }
        }
    }
}

// Export class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardCharts;
} else {
    window.DashboardCharts = DashboardCharts;
}
