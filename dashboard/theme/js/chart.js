/**
 * WEBSITE: https://themefisher.com
 * TWITTER: https://twitter.com/themefisher
 * FACEBOOK: https://www.facebook.com/themefisher
 * GITHUB: https://github.com/themefisher/
 */

/* ====== Index ======
1. SPLINA AREA CHART 01
2. SPLINA AREA CHART 02
3. SPLINA AREA CHART 03
4. SPLINA AREA CHART 04
5. MIXED CHART 01
6. RADIAL BAR CHART 01
7.1 HORIZONTAL BAR CHART
7.2 HORIZONTAL BAR CHART2
8.1 TABLE SMALL BAR CHART 01
8.2 TABLE SMALL BAR CHART 02
8.3 TABLE SMALL BAR CHART 03
8.4 TABLE SMALL BAR CHART 04
8.5 TABLE SMALL BAR CHART 05
8.6 TABLE SMALL BAR CHART 06
8.7 TABLE SMALL BAR CHART 07
8.8 TABLE SMALL BAR CHART 08
8.9 TABLE SMALL BAR CHART 09
8.10 TABLE SMALL BAR CHART 10
8.11 TABLE SMALL BAR CHART 11
8.12 TABLE SMALL BAR CHART 12
8.13 TABLE SMALL BAR CHART 13
8.14 TABLE SMALL BAR CHART 14
8.15 TABLE SMALL BAR CHART 15
9.1 STATUS SMALL BAR CHART 01
9.2 STATUS SMALL BAR CHART 02
9.3 STATUS SMALL BAR CHART 03
10.1 LINE CHART 01
10.2 LINE CHART 02
10.3 LINE CHART 03
10.4 LINE CHART 04
11.1 BAR CHART LARGE 01
11.2 BAR CHART LARGE 02
12.1 DONUT CHART 01
12.2 DONUT CHART 02
13. PIE CHART
14. RADER CHART
15.1 ARIA CHART 01

====== End ======*/

"use strict";

/*======== 1. SPLINA AREA CHART 01 ========*/
var splinaArea1 = document.querySelector("#spline-area-1");
if (splinaArea1 !== null) {
  var splinaAreaOptions1 = {
    chart: {
      id: "spline-area-1",
      group: "social",
      height: 135,
      width: "100%",
      background: "#fd5190",
      type: "area",
      sparkline: {
        enabled: true,
      },
    },
    yaxis: {
      labels: {
        minWidth: 40,
      },
    },
    stroke: {
      width: 2,
    },
    colors: ["rgba(255, 255, 255, .6)"],
    fill: {
      type: "gradient",
      gradient: {
        shade: "light",
        shadeIntensity: 1,
        opacityFrom: 0.3,
        opacityTo: 0.3,
        stops: [0, 50, 100],
      },
    },

    tooltip: {
      theme: "dark",
      marker: {
        show: false,
      },
      x: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },

    series: [
      {
        data: [0, 15, 18, 20, 16, 17, 23, 17, 25],
      },
    ],
  };
  var randerSplinaArea1 = new ApexCharts(splinaArea1, splinaAreaOptions1);
  randerSplinaArea1.render();
}

//   /*======== 2. SPLINA AREA CHART 02 ========*/
var splinaArea2 = document.querySelector("#spline-area-2");
if (splinaArea2 !== null) {
  var splinaAreaOptions2 = {
    chart: {
      id: "spline-area-1",
      group: "social",
      height: 135,
      width: "100%",
      background: "#46c79e",
      type: "area",
      sparkline: {
        enabled: true,
      },
    },
    yaxis: {
      labels: {
        minWidth: 40,
      },
    },
    stroke: {
      width: 2,
    },
    colors: ["#ffffff"],
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.7,
        opacityTo: 0.3,
        stops: [0, 90, 100],
      },
    },
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },

    series: [
      {
        data: [0, 4, 6, 14, 8, 10, 17, 20, 16],
      },
    ],
  };
  var randerSplinaArea2 = new ApexCharts(splinaArea2, splinaAreaOptions2);
  randerSplinaArea2.render();
}

//   /*======== 3. SPLINA AREA CHART 03 ========*/
var splinaArea3 = document.querySelector("#spline-area-3");
if (splinaArea3 !== null) {
  var splinaAreaOptions3 = {
    chart: {
      id: "spline-area-3",
      group: "social",
      height: 135,
      width: "100%",
      background: "#9e6de0",
      type: "area",
      sparkline: {
        enabled: true,
      },
    },
    yaxis: {
      labels: {
        minWidth: 40,
      },
    },
    stroke: {
      width: 2,
    },
    colors: ["#ffffff"],
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.7,
        opacityTo: 0.3,
        stops: [0, 90, 100],
      },
    },
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },

    series: [
      {
        data: [0, 8, 20, 14, 17, 12, 14, 8, 5],
      },
    ],
  };
  var randerSplinaArea3 = new ApexCharts(splinaArea3, splinaAreaOptions3);
  randerSplinaArea3.render();
}

// /*======== 4. SPLINA AREA CHART 04 ========*/
var splinaArea4 = document.querySelector("#spline-area-4");
if (splinaArea4 !== null) {
  var splinaAreaOptions4 = {
    chart: {
      id: "spline-area-3",
      group: "social",
      height: 135,
      width: "100%",
      background: "#6696fe",
      type: "area",
      sparkline: {
        enabled: true,
      },
    },
    yaxis: {
      labels: {
        minWidth: 40,
      },
    },
    stroke: {
      width: 2,
    },
    colors: ["#ffffff"],
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.7,
        opacityTo: 0.3,
        stops: [0, 90, 100],
      },
    },
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },

    series: [
      {
        data: [0, 3, 8, 15, 20, 10, 12, 10, 5],
      },
    ],
  };
  var randerSplinaArea4 = new ApexCharts(splinaArea4, splinaAreaOptions4);
  randerSplinaArea4.render();
}

//   /*======== 5. MIXED CHART 01 ========*/
var mixedChart1 = document.querySelector("#mixed-chart-1");
if (mixedChart1 !== null) {
  var mixedOptions1 = {
    chart: {
      height: 370,
      type: "bar",
      toolbar: {
        show: false,
      },
    },
    colors: ["#9e6de0", "#faafca", "#f2e052"],
    legend: {
      show: true,
      position: "top",
      horizontalAlign: "right",
      markers: {
        width: 20,
        height: 5,
        radius: 0,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        columnWidth: "50%",
        barHeight: "10%",
        distributed: false,
      },
    },
    dataLabels: {
      enabled: false,
    },

    stroke: {
      show: true,
      width: 2,
      curve: "smooth",
    },

    series: [
      {
        name: "Income",
        type: "column",
        data: [44, 55, 57, 56, 61, 58, 63, 60, 66, 55, 47, 67],
      },
      {
        name: "Expenses",
        type: "column",
        data: [76, 85, 101, 98, 87, 100, 91, 40, 94, 50, 47, 55],
      },
      {
        name: "profit",
        data: [50, 40, 64, 87, -15, 104, 63, 42, 32, 60, 78, 25],
        type: "line",
      },
    ],

    xaxis: {
      categories: [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
      ],

      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
      crosshairs: {
        width: 40,
      },
    },

    fill: {
      opacity: 1,
    },

    tooltip: {
      shared: true,
      intersect: false,
      followCursor: true,
      fixed: {
        enabled: false,
      },
      x: {
        show: false,
      },
      y: {
        title: {
          formatter: function (seriesName) {
            return seriesName;
          },
        },
      },
    },
  };

  var randerMixedChart1 = new ApexCharts(mixedChart1, mixedOptions1);
  randerMixedChart1.render();
}

/*======== 6. RADIAL BAR CHART 01 ========*/
var radialBarChart1 = document.querySelector("#radial-bar-chart-1");
if (radialBarChart1 !== null) {
  var radialBarOptions1 = {
    chart: {
      width: "100%",
      type: "radialBar",
      height: 345,
    },
    plotOptions: {
      radialBar: {
        size: 100,
        hollow: {
          size: "60%",
        },
        dataLabels: {
          show: true,
          name: {
            show: true,
            fontSize: "14px",
            fontFamily: undefined,
            color: "#222",
          },
          value: {
            show: true,
            fontSize: "16px",
            fontFamily: undefined,
            color: undefined,
            offsetY: 16,
            formatter: function () {
              return "";
            },
          },
        },
      },
    },
    fill: {
      type: "solid",
      colors: "#9e6de0",
    },
    series: [70],
    labels: ["Yearly Revenue"],
  };

  var randerRadialBar1 = new ApexCharts(radialBarChart1, radialBarOptions1);
  randerRadialBar1.render();
}

/*======== 7. HORIZONTAL BAR CHART ========*/
var horBarChart1 = document.querySelector("#horizontal-bar-chart");
if (horBarChart1 !== null) {
  var horBarChartOptions = {
    chart: {
      height: 325,
      type: "bar",
      toolbar: {
        show: false,
      },
      stacked: true,
    },
    colors: ["#9e6de0", "#faafca"],
    plotOptions: {
      bar: {
        horizontal: true,
        barHeight: "20%",
        distributed: true,
      },
    },
    dataLabels: {
      enabled: false,
    },
    series: [
      {
        data: [50, 45, 38, 27, 33, 19],
      },
    ],

    xaxis: {
      categories: [
        "India",
        "USA",
        "Canada",
        "Russia",
        "Austrolia",
        "Argentina",
      ],
      labels: {
        formatter: function (categories) {
          return categories;
        },
      },
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
      y: {
        title: {
          formatter: (seriesName) => "Sales",
        },
      },
    },
  };

  var randerHorBarChart1 = new ApexCharts(horBarChart1, horBarChartOptions);
  randerHorBarChart1.render();
}

/*======== 7. HORIZONTAL BAR CHART2 ========*/
var horBarChart2 = document.querySelector("#horizontal-bar-chart2");
if (horBarChart2 !== null) {
  var options = {
    chart: {
      height: 350,
      type: "bar",
      toolbar: {
        show: false,
      },
    },
    colors: ["#9e6de0", "#faafca"],
    plotOptions: {
      bar: {
        horizontal: true,
        barHeight: "50%",
        dataLabels: {
          position: "top",
        },
      },
    },
    legend: {
      show: true,
      position: "top",
      horizontalAlign: "right",
      markers: {
        width: 20,
        height: 5,
        radius: 0,
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 1,
      colors: ["#fff"],
    },
    series: [
      {
        data: [44, 55, 41, 64, 22, 43, 21],
      },
      {
        data: [53, 32, 33, 52, 13, 44, 32],
      },
    ],
    xaxis: {
      categories: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
      y: {
        title: {
          formatter: (seriesName) => "Sales",
        },
      },
    },
  };

  var chart = new ApexCharts(horBarChart2, options);

  chart.render();
}

/*======== 8.1 TABLE SMALL BAR CHART 01  ========*/
var tableSmBarChart1 = document.querySelector("#tbl-chart-01");
if (tableSmBarChart1 !== null) {
  var tableSmBarChartOption1 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [20, 30, 40, 50, 40, 25, 52, 25, 45, 25],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart1 = new ApexCharts(
    tableSmBarChart1,
    tableSmBarChartOption1
  );
  randerTblSmChart1.render();
}

/*======== 8.2 TABLE SMALL BAR CHART 02 ========*/
var tableSmBarChart2 = document.querySelector("#tbl-chart-02");
if (tableSmBarChart2 !== null) {
  var tableSmBarChartOption = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [25, 55, 35, 45, 66, 20, 50, 20, 50, 20],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart2 = new ApexCharts(
    tableSmBarChart2,
    tableSmBarChartOption
  );
  randerTblSmChart2.render();
}

/*======== 8.3 TABLE SMALL BAR CHART 03 ========*/
var tableSmBarChart3 = document.querySelector("#tbl-chart-03");
if (tableSmBarChart3 !== null) {
  var tableSmBarChartOption3 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [10, 30, 60, 15, 50, 45, 36, 17, 29, 65],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart3 = new ApexCharts(
    tableSmBarChart3,
    tableSmBarChartOption3
  );
  randerTblSmChart3.render();
}

/*======== 8.4 TABLE SMALL BAR CHART 04 ========*/
var tableSmBarChart4 = document.querySelector("#tbl-chart-04");
if (tableSmBarChart4 !== null) {
  var tableSmBarChartOption4 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [17, 50, 35, 58, 65, 15, 30, 17, 25, 42],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart4 = new ApexCharts(
    tableSmBarChart4,
    tableSmBarChartOption4
  );
  randerTblSmChart4.render();
}

/*======== 8.5 TABLE SMALL BAR CHART 05 ========*/
var tableSmBarChart5 = document.querySelector("#tbl-chart-05");
if (tableSmBarChart5 !== null) {
  var tableSmBarChartOption5 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [15, 42, 65, 49, 41, 29, 16, 45, 19, 17],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart5 = new ApexCharts(
    tableSmBarChart5,
    tableSmBarChartOption5
  );
  randerTblSmChart5.render();
}

/*======== 8.6 TABLE SMALL BAR CHART 06 ========*/
var tableSmBarChart6 = document.querySelector("#tbl-chart-06");
if (tableSmBarChart6 !== null) {
  var tableSmBarChartOption6 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [5, 32, 51, 43, 60, 19, 26, 35, 27, 17],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart6 = new ApexCharts(
    tableSmBarChart6,
    tableSmBarChartOption6
  );
  randerTblSmChart6.render();
}

/*======== 8.7 TABLE SMALL BAR CHART 07 ========*/
var tableSmBarChart7 = document.querySelector("#tbl-chart-07");
if (tableSmBarChart7 !== null) {
  var tableSmBarChartOption7 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [15, 42, 65, 49, 41, 29, 16, 45, 19, 17],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart7 = new ApexCharts(
    tableSmBarChart7,
    tableSmBarChartOption7
  );
  randerTblSmChart7.render();
}

/*======== 8.8 TABLE SMALL BAR CHART 08 ========*/
var tableSmBarChart8 = document.querySelector("#tbl-chart-08");
if (tableSmBarChart8 !== null) {
  var tableSmBarChartOption8 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [8, 25, 35, 18, 65, 52, 20, 35, 19, 9],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart8 = new ApexCharts(
    tableSmBarChart8,
    tableSmBarChartOption8
  );
  randerTblSmChart8.render();
}

/*======== 8.9 TABLE SMALL BAR CHART 09 ========*/
var tableSmBarChart9 = document.querySelector("#tbl-chart-09");
if (tableSmBarChart9 !== null) {
  var tableSmBarChartOption9 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [20, 32, 40, 19, 65, 19, 26, 23, 37, 20],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart9 = new ApexCharts(
    tableSmBarChart9,
    tableSmBarChartOption9
  );
  randerTblSmChart9.render();
}

/*======== 8.10 TABLE SMALL BAR CHART 10 ========*/
var tableSmBarChart10 = document.querySelector("#tbl-chart-10");
if (tableSmBarChart10 !== null) {
  var tableSmBarChartOption10 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [9, 25, 61, 18, 38, 26, 19, 28, 50, 40],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart10 = new ApexCharts(
    tableSmBarChart10,
    tableSmBarChartOption10
  );
  randerTblSmChart10.render();
}

/*======== 8.11 TABLE SMALL BAR CHART 11 ========*/
var tableSmBarChart11 = document.querySelector("#tbl-chart-11");
if (tableSmBarChart11 !== null) {
  var tableSmBarChartOption11 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [9, 42, 17, 35, 50, 52, 45, 65, 29, 38],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };
  var randerTblSmChart11 = new ApexCharts(
    tableSmBarChart11,
    tableSmBarChartOption11
  );
  randerTblSmChart11.render();
}

/*======== 8.12 TABLE SMALL BAR CHART 12 ========*/
var tableSmBarChart12 = document.querySelector("#tbl-chart-12");
if (tableSmBarChart12 !== null) {
  var tableSmBarChartOption12 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [15, 42, 65, 49, 41, 29, 16, 45, 19, 17],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart12 = new ApexCharts(
    tableSmBarChart12,
    tableSmBarChartOption12
  );
  randerTblSmChart12.render();
}

/*======== 8.13 TABLE SMALL BAR CHART 13 ========*/
var tableSmBarChart13 = document.querySelector("#tbl-chart-13");
if (tableSmBarChart13 !== null) {
  var tableSmBarChartOption13 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [26, 17, 9, 30, 41, 55, 63, 45, 19, 16],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart13 = new ApexCharts(
    tableSmBarChart13,
    tableSmBarChartOption13
  );
  randerTblSmChart13.render();
}

/*======== 8.14 TABLE SMALL BAR CHART 14 ========*/
var tableSmBarChart14 = document.querySelector("#tbl-chart-14");
if (tableSmBarChart14 !== null) {
  var tableSmBarChartOption14 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [15, 42, 65, 49, 41, 29, 16, 45, 19, 17],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart14 = new ApexCharts(
    tableSmBarChart14,
    tableSmBarChartOption14
  );
  randerTblSmChart14.render();
}

/*======== 8.15 TABLE SMALL BAR CHART 15 ========*/
var tableSmBarChart15 = document.querySelector("#tbl-chart-15");
if (tableSmBarChart15 !== null) {
  var tableSmBarChartOption15 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "rounded",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [9, 19, 46, 25, 30, 15, 27, 18, 65, 50],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#faafca",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerTblSmChart15 = new ApexCharts(
    tableSmBarChart15,
    tableSmBarChartOption15
  );
  randerTblSmChart15.render();
}
/*======== 9.1 STATUS SMALL BAR CHART 01  ========*/
var statusSmBarChart1 = document.querySelector("#status-sm-chart-01");
if (statusSmBarChart1 !== null) {
  var statusSmBarChartOption1 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "flat",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [20, 30, 40, 50, 40, 25, 52, 25, 45, 25],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#9e6de0",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerStatusSmChart1 = new ApexCharts(
    statusSmBarChart1,
    statusSmBarChartOption1
  );
  randerStatusSmChart1.render();
}

/*======== 9.2 STATUS SMALL BAR CHART 02 ========*/
var statusSmBarChart2 = document.querySelector("#status-sm-chart-02");
if (statusSmBarChart2 !== null) {
  var statusSmBarChartOption2 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "flat",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [25, 55, 35, 45, 66, 20, 50, 20, 50, 20],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#46c79e",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerStatusSmChart2 = new ApexCharts(
    statusSmBarChart2,
    statusSmBarChartOption2
  );
  randerStatusSmChart2.render();
}

/*======== 9.3 STATUS SMALL BAR CHART 03 ========*/
var statusSmBarChart3 = document.querySelector("#status-sm-chart-03");
if (statusSmBarChart3 !== null) {
  var statusSmBarChartOption3 = {
    chart: {
      height: 40,
      width: "100px",
      type: "bar",
      sparkline: {
        enabled: true,
      },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "flat",
        columnWidth: "65%",
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        data: [10, 30, 60, 15, 50, 45, 36, 17, 29, 65],
      },
    ],
    fill: {
      opacity: 1,
    },
    colors: "#04c7e0",
    tooltip: {
      followCursor: false,
      theme: "dark",
      x: {
        show: false,
      },
      marker: {
        show: false,
      },
      y: {
        title: {
          formatter: function () {
            return "";
          },
        },
      },
    },
  };

  var randerStatusSmChart3 = new ApexCharts(
    statusSmBarChart3,
    statusSmBarChartOption3
  );
  randerStatusSmChart3.render();
}

/*======== 10.1 LINE CHART 01 ========*/
var lineChart1 = document.querySelector("#line-chart-1");
if (lineChart1 !== null) {
  var lineChartOption1 = {
    chart: {
      height: 350,
      type: "line",
      toolbar: {
        show: false,
      },
    },
    stroke: {
      width: [2, 3],
      curve: "smooth",
      dashArray: [0, 5],
    },
    plotOptions: {
      horizontal: false,
    },
    colors: ["#9e6de0", "#fec400"],
    series: [
      {
        data: [6, 10, 8, 20, 15, 6, 21],
      },
      {
        data: [8, 6, 15, 10, 25, 8, 32],
      },
    ],
    labels: [
      "04 jan",
      "05 jan",
      "06 jan",
      "07 jan",
      "08 jan",
      "09 jan",
      "10 jan",
    ],
    markers: {
      size: [5, 0],
    },
    xaxis: {
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    tooltip: {
      theme: "dark",
      fixed: {
        enabled: false,
      },
      x: {
        show: false,
      },
      y: {
        title: {
          formatter: (labels) => labels,
        },
      },
      marker: {
        show: true,
      },
    },
    legend: {
      show: false,
    },
  };
  var randerLineChart1 = new ApexCharts(lineChart1, lineChartOption1);
  randerLineChart1.render();
}

/*======== 10.2 LINE CHART 02 ========*/
var lineChart2 = document.querySelector("#line-chart-2");
if (lineChart2 !== null) {
  var lineChartOption2 = {
    chart: {
      height: 350,
      type: "line",
      toolbar: {
        show: false,
      },
    },
    stroke: {
      width: [2, 3],
      curve: "smooth",
      dashArray: [0, 5],
    },
    plotOptions: {
      horizontal: false,
    },

    colors: ["#9e6de0", "#fec400"],
    series: [
      {
        data: [8, 15, 2, 12, 16, 25, 17],
      },
      {
        data: [5, 17, 12, 20, 11, 18, 12],
      },
    ],
    labels: [
      "04 jan",
      "05 jan",
      "06 jan",
      "07 jan",
      "08 jan",
      "09 jan",
      "10 jan",
    ],
    markers: {
      size: [5, 0],
    },
    xaxis: {
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    tooltip: {
      theme: "dark",
      shared: true,
      intersect: false,
      fixed: {
        enabled: false,
      },
      x: {
        show: false,
      },
      marker: {
        show: true,
      },
    },
    legend: {
      show: false,
    },
  };

  var randerLineChart2 = new ApexCharts(lineChart2, lineChartOption2);
  randerLineChart2.render();
}

/*======== 10.3 LINE CHART 03 ========*/
var lineChart3 = document.querySelector("#line-chart-3");
if (lineChart3 !== null) {
  var lineChartOption3 = {
    chart: {
      height: 350,
      type: "line",
      toolbar: {
        show: false,
      },
    },
    stroke: {
      width: [2, 3],
      curve: "smooth",
      dashArray: [0, 5],
    },
    plotOptions: {
      horizontal: false,
    },

    colors: ["#9e6de0", "#fec400"],
    series: [
      {
        data: [3, 9, 12, 24, 14, 11, 26],
      },
      {
        data: [6, 14, 18, 9, 22, 6, 17],
      },
    ],
    labels: [
      "04 jan",
      "05 jan",
      "06 jan",
      "07 jan",
      "08 jan",
      "09 jan",
      "10 jan",
    ],
    markers: {
      size: [5, 0],
    },
    xaxis: {
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    tooltip: {
      theme: "dark",
      shared: true,
      intersect: false,
      fixed: {
        enabled: false,
      },
      x: {
        show: false,
      },
      marker: {
        show: true,
      },
    },
    legend: {
      show: false,
    },
  };

  var randerLineChart3 = new ApexCharts(lineChart3, lineChartOption3);
  randerLineChart3.render();
}

/*======== 10.4 LINE CHART 04 ========*/
var lineChart4 = document.querySelector("#line-chart-4");
if (lineChart4 !== null) {
  var lineChartOption4 = {
    chart: {
      height: 350,
      type: "line",
      toolbar: {
        show: false,
      },
    },
    stroke: {
      width: [2, 3],
      curve: "smooth",
      dashArray: [0, 5],
    },
    plotOptions: {
      horizontal: false,
    },

    colors: ["#9e6de0", "#fec400"],

    legend: {
      show: true,
      position: "top",
      horizontalAlign: "right",
      markers: {
        width: 20,
        height: 5,
        radius: 0,
      },
    },
    series: [
      {
        data: [6, 10, 8, 20, 15, 6, 21],
      },
      {
        data: [8, 6, 15, 10, 25, 8, 32],
      },
    ],
    labels: [
      "04 jan",
      "05 jan",
      "06 jan",
      "07 jan",
      "08 jan",
      "09 jan",
      "10 jan",
    ],
    markers: {
      size: [5, 0],
    },
    xaxis: {
      axisBorder: {
        show: false,
      },
      axisTicks: {
        show: false,
      },
    },
    tooltip: {
      theme: "dark",
      shared: true,
      intersect: false,
      fixed: {
        enabled: false,
      },
      x: {
        show: false,
      },
      y: {
        title: {
          formatter: (labels) => labels,
        },
      },
      marker: {
        show: true,
      },
    },
  };
  var randerLineChart4 = new ApexCharts(lineChart4, lineChartOption4);
  randerLineChart4.render();
}

/*======== 11.1 BAR CHART LARGE 01 ========*/
var barChartLg1 = document.querySelector("#barchartlg1");
if (barChartLg1 !== null) {
  var barChartOptions1 = {
    chart: {
      height: 275,
      type: "bar",
      toolbar: {
        show: false,
      },
    },
    colors: ["#9e6de0", "#faafca", "#46c79e"],
    plotOptions: {
      bar: {
        horizontal: false,
        endingShape: "flat",
        columnWidth: "55%",
      },
    },
    legend: {
      position: "bottom",
      horizontalAlign: "left",
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      show: true,
      width: 2,
      colors: ["transparent"],
    },
    series: [
      {
        name: "Referral",
        data: [76, 85, 79, 88, 87, 65],
      },
      {
        name: "Direct",
        data: [44, 55, 57, 56, 61, 58],
      },
      {
        name: "Organic",
        data: [35, 41, 36, 26, 45, 48],
      },
    ],
    xaxis: {
      categories: ["4 Jan", "5 Jan", "6 Jan", "7 Jan", "8 Jan", "9 Jan"],
    },
    yaxis: {
      show: false,
    },
    fill: {
      opacity: 1,
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
      y: {
        formatter: function (val) {
          return "$ " + val;
        },
      },
      marker: {
        show: true,
      },
    },
  };
  var randerBarChartLg1 = new ApexCharts(barChartLg1, barChartOptions1);
  randerBarChartLg1.render();

  var items = document.querySelectorAll(
    "#user-acquisition .nav-underline-active-primary .nav-item"
  );
  items.forEach(function (item, index) {
    item.addEventListener("click", function () {
      if (index === 0) {
        randerBarChartLg1.updateSeries([
          {
            name: "Referral",
            data: [76, 85, 79, 88, 87, 65],
          },
          {
            name: "Direct",
            data: [44, 55, 57, 56, 61, 58],
          },
          {
            name: "Organic",
            data: [35, 41, 36, 26, 45, 48],
          },
        ]);
      } else if (index === 1) {
        randerBarChartLg1.updateSeries([
          {
            name: "iamabdus.com/referral",
            data: [66, 50, 35, 52, 52, 45],
          },
          {
            name: "github.com/referral",
            data: [49, 59, 75, 66, 15, 20],
          },
          {
            name: "(direct)/(none)",
            data: [55, 41, 65, 61, 53, 87],
          },
        ]);
      } else if (index === 2) {
        randerBarChartLg1.updateSeries([
          {
            name: "iamabdus.com",
            data: [64, 64, 58, 45, 77, 53],
          },
          {
            name: "tafcoder.com",
            data: [85, 25, 17, 12, 74, 15],
          },
          {
            name: "github.com",
            data: [51, 48, 53, 47, 55, 63],
          },
        ]);
      }
    });
  });
}

/*======== 11.2 BAR CHART LARGE 02 ========*/
var barChartLg2 = document.querySelector("#barchartlg2");
if (barChartLg2 !== null) {
  var trigoStrength = 3;
  var iteration = 11;

  function getRandom() {
    var i = iteration;
    return (
      (Math.sin(i / trigoStrength) * (i / trigoStrength) +
        i / trigoStrength +
        1) *
      (trigoStrength * 2)
    );
  }

  function getRangeRandom(yrange) {
    return (
      Math.floor(Math.random() * (yrange.max - yrange.min + 1)) + yrange.min
    );
  }

  function generateMinuteWiseTimeSeries(baseval, count, yrange) {
    var i = 0;
    var series = [];
    while (i < count) {
      var x = baseval;
      var y =
        (Math.sin(i / trigoStrength) * (i / trigoStrength) +
          i / trigoStrength +
          1) *
        (trigoStrength * 2);

      series.push([x, y]);
      baseval += 300000;
      i++;
    }
    return series;
  }

  var optionsColumn = {
    chart: {
      height: 315,
      type: "bar",
      toolbar: {
        show: false,
      },
      animations: {
        enabled: true,
        easing: "linear",
        dynamicAnimation: {
          speed: 1000,
        },
      },

      events: {
        animationEnd: function (chartCtx) {
          const newData = chartCtx.w.config.series[0].data.slice();
          newData.shift();
          window.setTimeout(function () {
            chartCtx.updateOptions(
              {
                series: [
                  {
                    name: "Load Average",
                    data: newData,
                  },
                ],
                xaxis: {
                  min: chartCtx.minX,
                  max: chartCtx.maxX,
                },
                subtitle: {
                  text: parseInt(
                    getRangeRandom({ min: 1, max: 20 })
                  ).toString(),
                },
              },
              false,
              false
            );
          }, 300);
        },
      },
      toolbar: {
        show: false,
      },
      zoom: {
        enabled: false,
      },
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      width: 0,
    },
    colors: "#9e6de0",
    series: [
      {
        name: "Load Average",
        data: generateMinuteWiseTimeSeries(
          new Date("12/12/2016 00:20:00").getTime(),
          12,
          {
            min: 10,
            max: 110,
          }
        ),
      },
    ],
    title: {
      text: "Ave Page views per minute",
      align: "left",
      offsetY: 35,
      style: {
        fontSize: "12px",
        color: "#8a909d",
      },
    },
    subtitle: {
      text: "20%",
      floating: false,
      align: "left",
      offsetY: 0,
      style: {
        fontSize: "22px",
        color: "#9e6de0",
      },
    },
    fill: {
      type: "solid",
      colors: "#9e6de0",
    },
    xaxis: {
      type: "datetime",
      range: 2700000,
    },
    legend: {
      show: false,
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
      y: {
        formatter: function (val) {
          return val;
        },
      },
      marker: {
        show: true,
      },
    },
  };

  var chartColumn = new ApexCharts(barChartLg2, optionsColumn);
  chartColumn.render();

  window.setInterval(function () {
    iteration++;

    chartColumn.updateSeries([
      {
        name: "Load Average",
        data: [
          ...chartColumn.w.config.series[0].data,
          [chartColumn.w.globals.maxX + 210000, getRandom()],
        ],
      },
    ]);
  }, 5000);
}

/*======== 12.1 DONUT CHART 01 ========*/
var donutChart1 = document.querySelector("#donut-chart-1");
if (donutChart1 !== null) {
  var donutChartOptions1 = {
    chart: {
      type: "donut",
      height: 270,
    },

    colors: ["#bb91f2", "#af81eb", "#9e6de0"],
    labels: ["Desktop", "Tablet", "Mobile"],
    series: [45, 30, 25],
    legend: {
      show: false,
    },
    dataLabels: {
      enabled: false,
    },
    tooltip: {
      y: {
        formatter: function (val) {
          return +val + "%";
        },
      },
    },
  };

  var randerDonutchart1 = new ApexCharts(donutChart1, donutChartOptions1);

  randerDonutchart1.render();
}

/*======== 12.2 DONUT CHART 02 ========*/
var donutChart2 = document.querySelector("#donut-chart-2");
if (donutChart2 !== null) {
  var donutChartOptions2 = {
    chart: {
      type: "donut",
      height: 330,
    },

    colors: ["#bb91f2", "#af81eb", "#9e6de0"],
    labels: ["Desktop", "Tablet", "Mobile"],
    series: [45, 30, 25],
    legend: {
      show: true,
      position: "top",
      horizontalAlign: "left",
      markers: {
        radius: 0,
      },
    },
    dataLabels: {
      enabled: false,
    },
    tooltip: {
      y: {
        formatter: function (val) {
          return +val + "%";
        },
      },
    },
  };

  var randerDonutchart2 = new ApexCharts(donutChart2, donutChartOptions2);

  randerDonutchart2.render();
}

/*======== 13. PIE CHART ========*/
var SimplePieChart = document.querySelector("#simple-pie-chart");
if (SimplePieChart !== null) {
  var simplePieChartOptions = {
    chart: {
      width: 350,
      type: "pie",
    },
    colors: ["#9e6de0", "#46c79e", "#fd5190"],
    labels: ["First Data", "Second Data", "Third Data"],
    legend: {
      position: "top",
      horizontalAlign: "left",
      markers: {
        radius: 0,
      },
    },
    series: [65, 25, 10],
  };

  var simpleplePieChartRander = new ApexCharts(
    SimplePieChart,
    simplePieChartOptions
  );

  simpleplePieChartRander.render();
}

/*======== 14. RADER CHART ========*/
var SimpleRaderChart = document.querySelector("#simple-rader-chart");
if (SimpleRaderChart !== null) {
  var options = {
    chart: {
      height: 350,
      type: "radar",
      sparkline: {
        enabled: true,
      },
    },
    labels: [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ],
    series: [
      {
        data: [80, 50, 30, 40, 100, 20, 80, 50, 30, 40, 100, 20],
      },
      {
        data: [20, 30, 40, 80, 20, 80, 20, 30, 40, 80, 20, 80],
      },
    ],
    tooltip: {
      enabled: false,
    },
  };

  var chart = new ApexCharts(SimpleRaderChart, options);

  chart.render();
}

/*======== 15.1 ARIA CHART 01 ========*/
var ariaChartExample = document.querySelector("#aria-chart");
if (ariaChartExample !== null) {
  var options = {
    chart: {
      height: 350,
      type: "area",
      toolbar: {
        show: false,
      },
    },
    colors: ["#9e6de0", "#faafca"],
    fill: {
      colors: ["#9e6de0", "#faafca"],
      type: "solid",
    },
    dataLabels: {
      enabled: false,
    },
    stroke: {
      curve: "smooth",
    },
    series: [
      {
        data: [31, 40, 28, 51, 42, 109, 100],
      },
      {
        data: [11, 32, 45, 32, 34, 52, 41],
      },
    ],
    legend: {
      show: false,
    },
    tooltip: {
      theme: "dark",
      x: {
        show: false,
      },
    },
  };

  var chart = new ApexCharts(ariaChartExample, options);

  chart.render();
}
