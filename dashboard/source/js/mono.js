/* ====== Index ======

1. SCROLLBAR SIDEBAR
2. MOBILE OVERLAY
3. SIDEBAR MENU
4. SIDEBAR TOGGLE FOR MOBILE
5. SIDEBAR TOGGLE FOR VARIOUS SIDEBAR LAYOUT
6. TODO LIST
7. RIGHT SIDEBAR
8. OFFCANVAS
9. DROPDOWN NOTIFY
10. REFRESS BUTTON
11. NAVBAR TRANSPARENT SCROLL
12. NAVBAR SEARCH
====== End ======*/

$(document).ready(function () {
  "use strict";

  /*======== 1. SCROLLBAR SIDEBAR ========*/

  /*======== 2. MOBILE OVERLAY ========*/
  if ($(window).width() < 768) {
    $(".sidebar-toggle").on("click", function () {
      $("body").css("overflow", "hidden");
      $("body").prepend('<div class="mobile-sticky-body-overlay"></div>');
    });

    $(document).on("click", ".mobile-sticky-body-overlay", function (e) {
      $(this).remove();
      $("#body")
        .removeClass("sidebar-mobile-in")
        .addClass("sidebar-mobile-out");
      $("body").css("overflow", "auto");
    });
  }

  /*======== 3. SIDEBAR MENU ========*/
  $(".sidebar .nav > .has-sub > a").click(function () {
    $(this).parent().siblings().removeClass("expand");
    $(this).parent().toggleClass("expand");
  });

  $(".sidebar .nav > .has-sub .has-sub > a").click(function () {
    $(this).parent().toggleClass("expand");
  });

  /*======== 4. SIDEBAR TOGGLE FOR MOBILE ========*/
  if ($(window).width() < 768) {
    $(document).on("click", ".sidebar-toggle", function (e) {
      e.preventDefault();
      var min = "sidebar-mobile-in",
        min_out = "sidebar-mobile-out",
        body = "#body";
      $(body).hasClass(min)
        ? $(body).removeClass(min).addClass(min_out)
        : $(body).addClass(min).removeClass(min_out);
    });
  }

  /*======== 5. SIDEBAR TOGGLE FOR VARIOUS SIDEBAR LAYOUT ========*/
  var body = $("#body");
  if ($(window).width() >= 768) {
    if (body.hasClass("sidebar-mobile-in sidebar-mobile-out")) {
      body.removeClass("sidebar-mobile-in sidebar-mobile-out");
    }

    window.isMinified = false;
    window.isCollapsed = false;

    $("#sidebar-toggler").on("click", function () {
      if (
        body.hasClass("sidebar-fixed-offcanvas") ||
        body.hasClass("sidebar-static-offcanvas")
      ) {
        $(this)
          .addClass("sidebar-offcanvas-toggle")
          .removeClass("sidebar-toggle");
        if (window.isCollapsed === false) {
          body.addClass("sidebar-collapse");
          window.isCollapsed = true;
          window.isMinified = false;
        } else {
          body.removeClass("sidebar-collapse");
          body.addClass("sidebar-collapse-out");
          setTimeout(function () {
            body.removeClass("sidebar-collapse-out");
          }, 300);
          window.isCollapsed = false;
        }
      }

      if (body.hasClass("sidebar-fixed") || body.hasClass("sidebar-static")) {
        $(this)
          .addClass("sidebar-toggle")
          .removeClass("sidebar-offcanvas-toggle");
        if (window.isMinified === false) {
          body
            .removeClass("sidebar-collapse sidebar-minified-out")
            .addClass("sidebar-minified");
          window.isMinified = true;
          window.isCollapsed = false;
        } else {
          body.removeClass("sidebar-minified");
          body.addClass("sidebar-minified-out");
          window.isMinified = false;
        }
      }
    });
  }

  if ($(window).width() >= 768 && $(window).width() < 992) {
    if (body.hasClass("sidebar-fixed") || body.hasClass("sidebar-static")) {
      body
        .removeClass("sidebar-collapse sidebar-minified-out")
        .addClass("sidebar-minified");
      window.isMinified = true;
    }
  }

  /*======== 6. TODO LIST ========*/

  function todoCheckAll() {
    var mdis = document.querySelectorAll(".todo-single-item .mdi");
    mdis.forEach(function (fa) {
      fa.addEventListener("click", function (e) {
        e.stopPropagation();
        e.target.parentElement.classList.toggle("finished");
      });
    });
  }

  if (document.querySelector("#todo")) {
    var list = document.querySelector("#todo-list"),
      todoInput = document.querySelector("#todo-input"),
      todoInputForm = todoInput.querySelector("form"),
      item = todoInputForm.querySelector("input");

    document.querySelector("#add-task").addEventListener("click", function (e) {
      e.preventDefault();
      todoInput.classList.toggle("d-block");
      item.focus();
    });

    todoInputForm.addEventListener("submit", function (e) {
      e.preventDefault();
      if (item.value.length <= 0) {
        return;
      }
      list.innerHTML =
        '<div class="todo-single-item d-flex flex-row justify-content-between">' +
        '<i class="mdi"></i>' +
        "<span>" +
        item.value +
        "</span>" +
        '<span class="badge badge-primary">Today</span>' +
        "</div>" +
        list.innerHTML;
      item.value = "";
      //Close input field
      todoInput.classList.toggle("d-block");
      todoCheckAll();
    });

    todoCheckAll();
  }

  /*======== 7. RIGHT SIDEBAR ========*/
  if ($(window).width() < 1025) {
    body.addClass("right-sidebar-toggoler-out");

    var btnRightSidebarToggler = $(".btn-right-sidebar-toggler");

    btnRightSidebarToggler.on("click", function () {
      if (!body.hasClass("right-sidebar-toggoler-out")) {
        body
          .addClass("right-sidebar-toggoler-out")
          .removeClass("right-sidebar-toggoler-in");
      } else {
        body
          .addClass("right-sidebar-toggoler-in")
          .removeClass("right-sidebar-toggoler-out");
      }
    });
  }

  var navRightSidebarLink = $(".nav-right-sidebar .nav-link");

  navRightSidebarLink.on("click", function () {
    if (!body.hasClass("right-sidebar-in")) {
      body.addClass("right-sidebar-in").removeClass("right-sidebar-out");
    } else if ($(this).hasClass("show")) {
      body.addClass("right-sidebar-out").removeClass("right-sidebar-in");
    }
  });

  var cardClosebutton = $(".card-right-sidebar .close");
  cardClosebutton.on("click", function () {
    body.removeClass("right-sidebar-in").addClass("right-sidebar-out");
  });

  /*======== 8. OFFCANVAS ========*/
  var offcanvasToggler = $(".offcanvas-toggler");
  var cardOffcanvas = $(".card-offcanvas");

  offcanvasToggler.on("click", function () {
    var offcanvasId = $(this).attr("data-offcanvas");
    cardOffcanvas.removeClass("active");
    $("#" + offcanvasId).addClass("active");
    $("#body").append('<div class="offcanvas-overlay"></div>');
  });

  /* Overlay Click*/
  $(document).on("click", ".offcanvas-overlay", function () {
    $(this).remove();
    cardOffcanvas.removeClass("active");
  });

  /*======== 9. DROPDOWN NOTIFY ========*/
  var dropdownToggle = $(".notify-toggler");
  var dropdownNotify = $(".dropdown-notify");

  if (dropdownToggle.length !== 0) {
    dropdownToggle.on("click", function () {
      if (!dropdownNotify.is(":visible")) {
        dropdownNotify.fadeIn(5);
      } else {
        dropdownNotify.fadeOut(5);
      }
    });

    $(document).mouseup(function (e) {
      if (
        !dropdownNotify.is(e.target) &&
        dropdownNotify.has(e.target).length === 0
      ) {
        dropdownNotify.fadeOut(5);
      }
    });
  }

  /*======== 10. REFRESS BUTTON ========*/
  var refressButton = $("#refress-button");
  if (refressButton !== 0) {
    refressButton.on("click", function () {
      $(this).addClass("mdi-spin");
      var $this = $(this);
      setTimeout(function () {
        $this.removeClass("mdi-spin");
      }, 3000);
    });
  }

  /*======== 11. NAVBAR TRANSPARENT SCROLL ========*/
  var body = $("#body");
  var navbar = $("#navbar");
  $(window).scroll(function () {
    if (
      body.hasClass("navbar-fixed") &&
      $(this).width() > 765 &&
      navbar.hasClass("navbar-transparent")
    ) {
      var scroll = $(window).scrollTop();

      if (scroll >= 10) {
        navbar.addClass("navbar-light").addClass("navbar-transparent");
      } else {
        navbar.removeClass("navbar-light").addClass("navbar-transparent");
      }
    }
  });

  /*======== 12. NAVBAR SEARCH ========*/
  var searchInput = $("#search-input");
  if (searchInput !== 0) {
    var inputSearch = $("#input-group-search");
    searchInput.focus(function () {
      $(".dropdown-menu-search").show();
      removeRadius();
      $(this).addClass("focus");
    });

    searchInput.focusout(function () {
      $(".dropdown-menu-search").hide();
      addRadius();
      $(this).removeClass("focus");
    });

    function removeRadius() {
      inputSearch.css({
        "border-bottom-left-radius": "0",
        "border-bottom-right-radius": "0",
      });
    }

    function addRadius() {
      inputSearch.css({
        "border-bottom-left-radius": ".5rem",
        "border-bottom-right-radius": ".5rem",
      });
    }

    window.displayBoxIndex = -1;
    searchInput.keyup(function (e) {
      if (e.keyCode == 40) {
        Navigate(1);
      }
      if (e.keyCode == 38) {
        Navigate(-1);
      }
      if (e.keyCode == 27) {
        $(".dropdown-menu-search").hide();
        addRadius();
      }
    });

    var Navigate = function (diff) {
      displayBoxIndex += diff;
      var oBoxCollection = $(".dropdown-menu-search .nav-item");
      if (displayBoxIndex >= oBoxCollection.length) displayBoxIndex = 0;
      if (displayBoxIndex < 0) displayBoxIndex = oBoxCollection.length - 1;
      var cssClass = "active";
      oBoxCollection
        .removeClass(cssClass)
        .eq(displayBoxIndex)
        .addClass(cssClass);
    };
  }

  hotkeys("/", function (event, handler) {
    switch (handler.key) {
      case "/":
        event.preventDefault();
        searchInput.focus();
        break;
    }
  });
});
