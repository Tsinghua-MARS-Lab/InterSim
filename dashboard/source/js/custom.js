/* ====== Index ======

1. SCROLLBAR CONTENT
2. TOOLTIPS AND POPOVER
3. JVECTORMAP HOME WORLD
4. JVECTORMAP USA REGIONS VECTOR MAP
5. COUNTRY SALES RANGS
6. JVECTORMAP HOME WORLD
7. CODE EDITOR
8. QUILL TEXT EDITOR
9. MULTIPLE SELECT
10. LOADING BUTTON
11. TOASTER
12. INFO BAR
13. PROGRESS BAR
14. DATA TABLE
15. OWL CAROUSEL

====== End ======*/

$(document).ready(function () {
  "use strict";

  /*======== 1. SCROLLBAR CONTENT ========*/

  /*======== 2. TOOLTIPS AND POPOVER ========*/
  $('[data-toggle="tooltip"]').tooltip({
    container: "body",
    template:
      '<div class="tooltip" role="tooltip"><div class="arrow"></div><div class="tooltip-inner"></div></div>',
  });
  $('[data-toggle="popover"]').popover();

  /*======== 3. JVECTORMAP HOME WORLD ========*/
  var homeWorld = $("#home-world");
  if (homeWorld.length != 0) {
    var colorData = {
      CA: 106,
      US: 166,
      RU: 166,
      AR: 166,
      AU: 120,
      IN: 106,
    };
    homeWorld.vectorMap({
      map: "world_mill",
      backgroundColor: "#fff",
      zoomOnScroll: false,
      regionStyle: {
        initial: {
          fill: "#cbccd4",
        },
      },
      series: {
        regions: [
          {
            values: colorData,
            scale: ["#9e6cdf", "#dfe0e4", "#f9aec9"],
          },
        ],
      },
    });
  }

  /*======== 4. JVECTORMAP USA REGIONS VECTOR MAP ========*/
  var usVectorMap = $("#us-vector-map-marker");
  if (usVectorMap.length != 0) {
    usVectorMap.vectorMap({
      map: "us_aea",
      backgroundColor: "#transparent",
      zoomOnScroll: false,
      regionStyle: {
        initial: {
          fill: "#eff0f5",
        },
      },
      markerStyle: {
        hover: {
          stroke: "transparent",
        },
      },
      markers: [
        {
          latLng: [39.55, -105.78],
          name: "Colorado",
          style: { fill: "#46c79e", stroke: "#46c79e" },
        },
        {
          latLng: [40.26, -86.13],
          name: "Indiana",
          style: { fill: "#fec402", stroke: "#fec402" },
        },
        {
          latLng: [43.8, -120.55],
          name: "Oregon",
          style: { fill: "#9e6de0", stroke: "#9e6de0" },
        },
      ],
    });
  }

  /*======== 5. COUNTRY SALES RANGS ========*/
  var countrySalesRange = $("#country-sales-range");
  if (countrySalesRange.length != 0) {
    var start = moment().subtract(29, "days");
    var end = moment();

    function cb(start, end) {
      $("#country-sales-range .date-holder").html(
        start.format("MMMM D, YYYY") + " - " + end.format("MMMM D, YYYY")
      );
    }

    countrySalesRange.daterangepicker(
      {
        startDate: start,
        endDate: end,
        opens: "left",
        ranges: {
          Today: [moment(), moment()],
          Yesterday: [
            moment().subtract(1, "days"),
            moment().subtract(1, "days"),
          ],
          "Last 7 Days": [moment().subtract(6, "days"), moment()],
          "Last 30 Days": [moment().subtract(29, "days"), moment()],
          "This Month": [moment().startOf("month"), moment().endOf("month")],
          "Last Month": [
            moment().subtract(1, "month").startOf("month"),
            moment().subtract(1, "month").endOf("month"),
          ],
        },
      },
      cb
    );

    cb(start, end);
  }
  var miniStatusRanges = $("#mini-status-range");

  if (miniStatusRanges.length != 0) {
    var start = moment().subtract(29, "days");
    var end = moment();

    function cb(start, end) {
      $("#mini-status-range .date-holder").html(
        start.format("MMMM D, YYYY") + " - " + end.format("MMMM D, YYYY")
      );
    }

    miniStatusRanges.daterangepicker(
      {
        startDate: start,
        endDate: end,
        opens: "left",
        ranges: {
          Today: [moment(), moment()],
          Yesterday: [
            moment().subtract(1, "days"),
            moment().subtract(1, "days"),
          ],
          "Last 7 Days": [moment().subtract(6, "days"), moment()],
          "Last 30 Days": [moment().subtract(29, "days"), moment()],
          "This Month": [moment().startOf("month"), moment().endOf("month")],
          "Last Month": [
            moment().subtract(1, "month").startOf("month"),
            moment().subtract(1, "month").endOf("month"),
          ],
        },
      },
      cb
    );

    cb(start, end);
  }

  /*======== 6. JVECTORMAP HOME WORLD ========*/
  var countryWithMarker = $("#world-country-with-marker");
  if (countryWithMarker.length != 0) {
    var colorData = {
      CA: 106,
      US: 166,
      RU: 166,
      AR: 166,
      AU: 120,
      IN: 106,
    };
    countryWithMarker.vectorMap({
      map: "world_mill",
      backgroundColor: "#fff",
      zoomOnScroll: false,
      regionStyle: {
        initial: {
          fill: "#cbccd4",
        },
      },
      series: {
        regions: [
          {
            values: colorData,
            scale: ["#9e6cdf", "#dfe0e4", "#f9aec9"],
          },
        ],
      },
      markers: [
        { latLng: [56.13, -106.34], name: "Vatican City" },
        { latLng: [37.09, -95.71], name: "Washington" },
        { latLng: [-14.23, -51.92], name: "Brazil" },
        { latLng: [17.6078, 8.0817], name: "Tuvalu" },
        { latLng: [47.14, 9.52], name: "Liechtenstein" },
        { latLng: [20.59, 78.96], name: "India" },
        { latLng: [61.52, 105.31], name: "Russia" },
      ],
    });
  }

  var usVectorMapWithoutMarker = $("#us-vector-map-without-marker");
  if (usVectorMapWithoutMarker.length != 0) {
    usVectorMapWithoutMarker.vectorMap({
      map: "us_aea",
      backgroundColor: "#transparent",
      zoomOnScroll: false,
      regionStyle: {
        initial: {
          fill: "#eff0f5",
        },
      },
      markerStyle: {
        hover: {
          stroke: "transparent",
        },
      },
    });
  }

  /*======== 7. CODE EDITOR ========*/
  var codeEditor = document.getElementById("code-editor");
  if (codeEditor) {
    var htmlCode = `<html style="color: green">
  <!-- this is a comment -->
  <head>"
    <title>HTML Example</title>
  </head>
  <body>
    The indentation tries to be <em>somewhat &quot;do what
    I mean&quot;</em>... but might not match your style.
  </body>
</html>`;

    var myCodeMirror = CodeMirror(codeEditor, {
      value: htmlCode,
      mode: "xml",
      extraKeys: { "Ctrl-Space": "autocomplete" },
      lineNumbers: true,
      indentWithTabs: true,
      lineWrapping: true,
    });
  }

  /*======== 8. QUILL TEXT EDITOR ========*/
  var quillHook = document.getElementById("editor");
  if (quillHook !== null) {
    var quill = new Quill(quillHook, {
      modules: {
        formula: false,
        syntax: false,
        toolbar: "#toolbar",
      },
      placeholder: "Enter Text ...",
      theme: "snow",
    });
  }

  /*======== 9. MULTIPLE SELECT ========*/
  var select2Multiple = $(".js-example-basic-multiple");
  if (select2Multiple.length != 0) {
    select2Multiple.select2();
  }
  var select2Country = $(".country");
  if (select2Country.length != 0) {
    select2Country.select2({
      minimumResultsForSearch: -1,
    });
  }

  /*======== 10. LOADING BUTTON ========*/
  var laddaButton = $(".ladda-button");
  if (laddaButton.length != 0) {
    Ladda.bind(".ladda-button", {
      timeout: 1000,
    });
  }

  /*======== 11. TOASTER ========*/
  var toaster = $("#toaster");
  function callToaster(positionClass) {
    toastr.options = {
      closeButton: true,
      debug: false,
      newestOnTop: false,
      progressBar: true,
      positionClass: positionClass,
      preventDuplicates: false,
      onclick: null,
      showDuration: "300",
      hideDuration: "1000",
      timeOut: "5000",
      extendedTimeOut: "1000",
      showEasing: "swing",
      hideEasing: "linear",
      showMethod: "fadeIn",
      hideMethod: "fadeOut",
    };
    toastr.success("Welcome to InterSim Dashboard", "Howdy!");
  }

  if (toaster.length != 0) {
    if (document.dir != "rtl") {
      callToaster("toast-top-right");
    } else {
      callToaster("toast-top-left");
    }
  }

  /*======== 12. INFO BAR ========*/
  var infoTeoaset = $(
    "#toaster-info, #toaster-success, #toaster-warning, #toaster-danger"
  );
  if (infoTeoaset !== null) {
    infoTeoaset.on("click", function () {
      toastr.options = {
        closeButton: true,
        debug: false,
        newestOnTop: false,
        progressBar: false,
        positionClass: "toast-top-right",
        preventDuplicates: false,
        onclick: null,
        showDuration: "3000",
        hideDuration: "1000",
        timeOut: "5000",
        extendedTimeOut: "1000",
        showEasing: "swing",
        hideEasing: "linear",
        showMethod: "fadeIn",
        hideMethod: "fadeOut",
      };
      var thisId = $(this).attr("id");
      if (thisId === "toaster-info") {
        toastr.info("Welcome to Mono", " Info message");
      } else if (thisId === "toaster-success") {
        toastr.success("Welcome to Mono", "Success message");
      } else if (thisId === "toaster-warning") {
        toastr.warning("Welcome to Mono", "Warning message");
      } else if (thisId === "toaster-danger") {
        toastr.error("Welcome to Mono", "Danger message");
      }
    });
  }

  /*======== 13. PROGRESS BAR ========*/
  NProgress.done();

  /*======== 14. DATA TABLE ========*/
  // var productsTable = $("#productsTable");
  // if (productsTable.length != 0) {
  //   productsTable.DataTable({
  //     info: false,
  //     lengthChange: false,
  //     searching: false,
  //     lengthMenu: [
  //       [20, 50, 100, -1],
  //       [20, 50, 100, "All"],
  //     ],
  //     scrollX: true,
  //     // order: [[6, "desc"]],
  //     columnDefs: [
  //       {
  //         orderable: false,
  //         targets: [0, 1, 2, 3, 4, 5, 6, 7],
  //       },
  //     ],
  //     language: {
  //       search: "_INPUT_",
  //       searchPlaceholder: "Search...",
  //     },
  //   });
  // }

  var leaderboardTable = $("#leaderboardTable");
  if (leaderboardTable.length != 0) {
    leaderboardTable.DataTable({
      info: false,
      lengthChange: false,
      searching: false,
      paging: false,
      lengthMenu: [
        [20, 50, 100, -1],
        [20, 50, 100, "All"],
      ],
      // scrollX: true,
      order: [[2, "desc"]],
      columnDefs: [
        {
          orderable: false,
          targets: [0, 1],
        },
      ]
    });
  }

  var leaderboardTable2 = $("#leaderboardTable2");
  if (leaderboardTable2.length != 0) {
    leaderboardTable2.DataTable({
      info: false,
      lengthChange: false,
      searching: false,
      paging: false,
      lengthMenu: [
        [20, 50, 100, -1],
        [20, 50, 100, "All"],
      ],
      // scrollX: true,
      order: [[2, "desc"]],
      columnDefs: [
        {
          orderable: false,
          targets: [0, 1],
        },
      ]
    });
  }

  var leaderboardTable3 = $("#leaderboardTable3");
  if (leaderboardTable3.length != 0) {
    leaderboardTable3.DataTable({
      info: false,
      lengthChange: false,
      searching: false,
      paging: false,
      lengthMenu: [
        [20, 50, 100, -1],
        [20, 50, 100, "All"],
      ],
      // scrollX: true,
      order: [[2, "desc"]],
      columnDefs: [
        {
          orderable: false,
          targets: [0, 1],
        },
      ]
    });
  }

  var leaderboardTable4 = $("#leaderboardTable4");
  if (leaderboardTable4.length != 0) {
    leaderboardTable4.DataTable({
      info: false,
      lengthChange: false,
      searching: false,
      paging: false,
      lengthMenu: [
        [20, 50, 100, -1],
        [20, 50, 100, "All"],
      ],
      // scrollX: true,
      order: [[2, "desc"]],
      columnDefs: [
        {
          orderable: false,
          targets: [0, 1],
        },
      ]
    });
  }

  var scenarioTable = $("#scenarioTable");
  if (scenarioTable.length != 0) {
    scenarioTable.DataTable({
      info: false,
      lengthChange: false,
      searching: false,
      paging: false,
      lengthMenu: [
        [20, 50, 100, -1],
        [20, 50, 100, "All"],
      ],
      // scrollX: true,
      order: [[1, "desc"]],
      columnDefs: [
        {
          orderable: false,
          targets: [0, -1],
        },
      ]
    });
  }

  var productSale = $("#product-sale");
  if (productSale.length != 0) {
    productSale.DataTable({
      info: false,
      paging: false,
      searching: false,
      scrollX: true,
      order: [[0, "asc"]],
      columnDefs: [
        {
          orderable: false,
          targets: [-1],
        },
      ],
    });
  }

  /*======== 15. OWL CAROUSEL ========*/
  var slideOnly = $(".slide-only");
  if (slideOnly.length != 0) {
    slideOnly.owlCarousel({
      items: 1,
      autoplay: true,
      loop: true,
      dots: false,
    });
  }

  var carouselWithControl = $(".carousel-with-control");
  if (carouselWithControl.length != 0) {
    carouselWithControl.owlCarousel({
      items: 1,
      autoplay: true,
      loop: true,
      dots: false,
      nav: true,
      navText: [
        '<i class="mdi mdi-chevron-left"></i>',
        '<i class="mdi mdi-chevron-right"></i>',
      ],
      center: true,
    });
  }

  var carouselWithIndicators = $(".carousel-with-indicators");
  if (carouselWithIndicators.length != 0) {
    carouselWithIndicators.owlCarousel({
      items: 1,
      autoplay: true,
      loop: true,
      nav: true,
      navText: [
        '<i class="mdi mdi-chevron-left"></i>',
        '<i class="mdi mdi-chevron-right"></i>',
      ],
      center: true,
    });
  }

  var caoruselWithCaptions = $(".carousel-with-captions");
  if (caoruselWithCaptions.length != 0) {
    caoruselWithCaptions.owlCarousel({
      items: 1,
      autoplay: true,
      loop: true,
      nav: true,
      navText: [
        '<i class="mdi mdi-chevron-left"></i>',
        '<i class="mdi mdi-chevron-right"></i>',
      ],
      center: true,
    });
  }

  var carouselUser = $(".carousel-user");
  if (carouselUser.length != 0) {
    carouselUser.owlCarousel({
      items: 4,
      margin: 80,
      autoplay: true,
      loop: true,
      nav: true,
      navText: [
        '<i class="mdi mdi-chevron-left"></i>',
        '<i class="mdi mdi-chevron-right"></i>',
      ],
      responsive: {
        0: {
          items: 1,
          margin: 0,
        },
        768: {
          items: 2,
        },
        1000: {
          items: 3,
        },
        1440: {
          items: 4,
        },
      },
    });
  }

  var carouselTestimonial = $(".carousel-testimonial");
  if (carouselTestimonial.length != 0) {
    carouselTestimonial.owlCarousel({
      items: 3,
      margin: 135,
      autoplay: false,
      loop: true,
      nav: true,
      navText: [
        '<i class="mdi mdi-chevron-left"></i>',
        '<i class="mdi mdi-chevron-right"></i>',
      ],
      responsive: {
        0: {
          items: 1,
          margin: 0,
        },
        768: {
          items: 1,
        },
        1000: {
          items: 2,
        },
        1440: {
          items: 3,
        },
      },
    });
  }

  /*======== 7. CIRCLE PROGRESS ========*/
  var circle = $(".circle");
  var gray = "#f5f6fa";

  if (circle.length != 0) {
    circle.circleProgress({
      lineCap: "round",
      startAngle: 4.8,
      emptyFill: [gray],
    });
  }
});
