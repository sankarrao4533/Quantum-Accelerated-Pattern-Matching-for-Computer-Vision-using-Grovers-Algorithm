(function () {
  if (typeof window.gsap === 'undefined') {
    return;
  }

  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (typeof window.ScrollTrigger !== 'undefined') {
    gsap.registerPlugin(ScrollTrigger);
  }

  function animateLanding() {
    if (reduceMotion) {
      return;
    }

    gsap.from('.navbar', {
      y: -24,
      opacity: 0,
      duration: 0.6,
      ease: 'power2.out'
    });

    gsap.from('.header h1', {
      y: 26,
      opacity: 0,
      duration: 0.7,
      ease: 'power2.out',
      delay: 0.08
    });

    gsap.from('.header p, .tech-badge', {
      y: 20,
      opacity: 0,
      duration: 0.58,
      stagger: 0.06,
      ease: 'power2.out',
      delay: 0.2
    });

    gsap.from('.step-card, .upload-card, .analyze-wrap', {
      y: 24,
      opacity: 0,
      duration: 0.64,
      stagger: 0.08,
      ease: 'power2.out',
      delay: 0.34
    });
  }

  function animateOnScroll() {
    if (reduceMotion || typeof gsap.plugins === 'undefined' || !gsap.plugins.scrollTrigger) {
      return;
    }

    setTimeout(function () {
      var groups = [
        '.steps-section .step-card',
        '.upload-section .upload-card',
        '.analyze-wrap',
        '.results-section .section-title',
        '.footer'
      ];

      groups.forEach(function (selector) {
        var nodes = document.querySelectorAll(selector);
        if (!nodes.length) {
          return;
        }

        nodes.forEach(function (node) {
          gsap.from(node, {
            y: 28,
            opacity: 0,
            duration: 0.7,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: node,
              start: 'top 85%',
              end: 'top 50%',
              toggleActions: 'play none none none',
              once: true,
              markers: false
            }
          });
        });
      });
    }, 100);
  }

  function animateResults() {
    var results = document.getElementById('resultsSection');
    if (!results) {
      return;
    }

    var observer = new MutationObserver(function () {
      if (!results.classList.contains('visible')) {
        return;
      }

      var targets = results.querySelectorAll(
        '#resultBanner, #statsGrid .stat-card, #confidenceSection .confidence-wrap, #gaugesGrid .gauge-card, .output-wrap, #thumbsRow .thumb-card, #chartsGrid .chart-card, #timingGrid .timing-card, #compareSection > *, #imageInfoGrid .image-info-card, #scoreTableWrap, #quantumInfo > *, #accordions .accordion'
      );

      if (!targets.length) {
        return;
      }

      if (typeof window.ScrollTrigger !== 'undefined') {
        ScrollTrigger.refresh();
      }

      gsap.fromTo(
        targets,
        { y: 18, opacity: 0 },
        {
          y: 0,
          opacity: 1,
          duration: 0.5,
          stagger: 0.035,
          ease: 'power2.out',
          overwrite: true
        }
      );
    });

    observer.observe(results, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ['class']
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      animateLanding();
      animateResults();
      animateOnScroll();
      setTimeout(function () {
        if (gsap.plugins && gsap.plugins.scrollTrigger) {
          gsap.plugins.scrollTrigger.refresh();
        }
      }, 200);
    });
  } else {
    animateLanding();
    animateResults();
    animateOnScroll();
    setTimeout(function () {
      if (gsap.plugins && gsap.plugins.scrollTrigger) {
        gsap.plugins.scrollTrigger.refresh();
      }
    }, 200);
  }
})();
