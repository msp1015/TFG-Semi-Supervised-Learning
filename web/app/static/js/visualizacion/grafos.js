let simulation, svg, container, link, node, clases;
let color;
let currentStep = 0;
let steps = dataSteps = [];
let maxiter = 0;
let predictions = {};
let selectedNode = null;
let initialColors = {};

function inicializarDatos(datos) {
    predictions = datos.predicciones;
    steps = datos.enlaces;
    dataSteps.push({
    nodes: datos.nodos,
    links: steps[0],
    });
    for (let i = 0; i < steps.length; i++) {
    dataSteps.push({
        nodes: datos.nodos,
        links: steps[i],
    });
    }
    clases = JSON.parse(datos.mapa);
    maxiter = steps.length;
    console.log(maxiter);
}



let intervalo = null;

/**
 *
 * Se encarga de la reproducción automática de la visualización al
 * pulsar dicho botón.
 *
 * Tiempo: 1 segundo y medio por iteración.
 *
 */
function reproducir(){
    if (!intervalo){
        document.getElementById("reproducir").innerHTML = "<i class='bi bi-stop-fill'></i>";
        intervalo = setInterval(function () {
            if (currentStep+1 >= maxiter){
                document.getElementById("reproducir").innerHTML = "<i class='bi bi-play-fill'></i>";
                clearInterval(intervalo);
                intervalo = null;
            }
            changeStep(1);
        }, 1500)
    } else {
        document.getElementById("reproducir").innerHTML = "<i class='bi bi-play-fill'></i>";
        clearInterval(intervalo);
        intervalo = null;
    }
}

function highlightNodosDeClase(clase) {
  node.each(function(d) {
    if (d.class === clase) {
      d3.select(this).attr("r", 12); // Aumenta el tamaño del nodo
    } else {
      d3.select(this).attr("r", 5); // Restablece el tamaño del nodo
    }
  });
}


function inicializarGrafo() {

    let margin = {top: 10, right: 0, bottom: 60, left: 45},
        width = 750 - margin.left - margin.right,
        height = 600 - margin.top - margin.bottom;

    color = d3.scaleOrdinal()
    .domain(Object.keys(clases))
    .range(d3.schemeSet3);

    let leyenda = document.getElementById("leyenda_visualizacion");

    leyenda.innerHTML = "";

    for (const clase of Object.keys(clases)) {
      let span = document.createElement("span");
      span.style.color = color(parseInt(clase));
      span.innerHTML = clases[clase];
      span.addEventListener('click', function() {
        highlightNodosDeClase(parseInt(clase));
      });
      leyenda.appendChild(span);
    }

    svg = d3.select("#visualizacion_principal")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .style("border", "3px solid rgba(0, 0, 0, 0.3)")
        .style("border-radius", "10px");

    container = svg.append('g')        
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .style("margin", "auto");

    link = container.append("g")
        .attr("stroke", "#999")
        .attr("stroke-opacity", 0) // Para no mostrar enlaces en la primera
        .selectAll("line");

    node = container.append("g")
        .attr("stroke", "#ccc")
        .attr("stroke-width", 1)
        .selectAll("circle");

    simulation = d3.forceSimulation()
        .nodes(dataSteps[0].nodes)
        .force("charge", d3.forceManyBody().strength(-100).theta(0.1).distanceMax(500))
        .force("x", d3.forceX(width / 2).strength(0.1))
        .force("y", d3.forceY(height / 2).strength(0.1))
        .force("link", d3.forceLink(dataSteps[0].links).id(d => d.id)) 
        .on("tick", () => {
          node
              .attr("cx", d => d.x)
              .attr("cy", d => d.y);

          link
              .attr("x1", d => d.source.x)
              .attr("y1", d => d.source.y)
              .attr("x2", d => d.target.x)
              .attr("y2", d => d.target.y);
        });

    
    const zoom = d3.zoom()
      .scaleExtent([0.5, 10])
      .extent([[0, 0], [width, height]])
      .on("zoom", updateChart);

    const initialScale = 0.60; 
    const initialTransform = d3.zoomIdentity
        .translate(margin.left + margin.right +  width / 2, margin.top + margin.bottom + height / 2) 
        .scale(initialScale)
        .translate(-width / 2, -height / 2);
    
    svg.call(zoom)
        .call(zoom.transform, initialTransform);

    document.querySelector("#reiniciar_zoom").addEventListener("click", function (){
        d3.select("#visualizacion_principal svg")
            .transition()
            .duration(750)
            .call(zoom.transform, initialTransform);
    });

    svg.on("click", () => {
        selectedNode = null;
        updateGraph();
    });

    document.getElementById('previt').addEventListener('click', () => changeStep(-1));
    document.getElementById('nextit').addEventListener('click', () => changeStep(1));
    document.getElementById('reproducir').addEventListener('click', () => reproducir());
}

function updateChart(event) {
    const transform = event.transform;
    container.attr("transform", transform);
}

function changeStep(direction) {
    currentStep += direction;
    if (currentStep < 0) currentStep = 0;
    if (currentStep >= dataSteps.length) currentStep = dataSteps.length - 1;
    // Asegura que se deshabilita el boton de resultados de inferencia
    if(currentStep < maxiter) {
      let button = document.getElementById('btn-inferencia');
      button.disabled = true;
      button.classList.add('disabled');
      $('#inferencia').hide();
      $('#fases_grafo').show(); 
      $('#btn-fases-grafo').addClass('active');
    }
    actualizaBarraProgreso(currentStep);
    updateGraph();
    muestraPaso(currentStep);
}

function muestraPaso(step) {
  for (let i = 0; i <= maxiter; i++) {
    document.getElementById(`step-${i}`).classList.remove('active');
  }
  document.getElementById(`step-${step}`).classList.add('active');
}

function actualizaBarraProgreso(step) {
    let progreso = (step / maxiter) * 100;
    document.getElementById('progreso').style.width = `${progreso}%`;
    document.getElementById("progreso").setAttribute("aria-valuenow", progreso.toString());
    document.getElementById("iteracion").innerHTML = traducir(nombreAlgoritmo + "_" + step.toString());
}

function updateGraph() {
    const currentData = dataSteps[currentStep];
    const nodes = currentData.nodes;
    const links = currentData.links;
  

    // Update nodes
    node = node.data(nodes, d => d.id)
      .join(enter => enter.append("circle")
          .attr("r", 5)
          .attr("fill",d => d.class === -1 ? '#808080' : color(d.class))
          .call(enter => enter.append("text"))
          .on("click", (event, d) => {
            if (currentStep > 0) { // Activar clics solo después del primer paso
                event.stopPropagation();
                selectedNode = d.id;
                updateGraph();
            }
            })
        );
    if (currentStep != dataSteps.length - 1){
      node.each(function(d) {
          d3.select(this)
          .attr("fill", d => d.class === -1 ? '#808080' : color(d.class))
      });
    }
   
    link = link.data(links, d => `${d.source}-${d.target}`)
      .join("line")
      .attr("stroke-opacity", currentStep === 0 ? 0 : 0.85)
      .attr("stroke-width", d => (selectedNode && (d.source.id === selectedNode || d.target.id === selectedNode)) ? 3 : 1);
    
    node.attr("r", d => (currentStep > 0 && selectedNode && (d.id === selectedNode 
        || links.some(link => (link.source.id === selectedNode && link.target.id === d.id) 
        || (link.target.id === selectedNode && link.source.id === d.id)))) ? 8 : 5);

    // Update simulation
    simulation.nodes(nodes);
    if (currentStep > 0) {
      simulation.force("link", d3.forceLink(links).id(d => d.id));
    }
    simulation.alpha(0.3).restart();

    // Arrastrar solo en el ultimo paso
    if (currentStep === dataSteps.length - 1) {
      node.call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));
    } else {
      node.on(".drag", null);
    }

    if (currentStep === dataSteps.length - 1 ) {
      var button = document.getElementById('inferir_etiq');
      button.disabled = false;
      button.classList.remove('disabled');
      document.getElementById('inferir_etiq').addEventListener('click', () => inferLabels());
    }

    // Asegurar que el botón de inferencia se deshabilite si no es el último paso
    if (currentStep < dataSteps.length - 1) {
      const inferButton = document.getElementById('inferir_etiq');
      if (inferButton) {
        var button = document.getElementById('inferir_etiq');
        button.disabled = true;
        button.classList.add('disabled');
      }
    }

    
}
function inferLabels() {
  var button = document.getElementById('btn-inferencia');
  button.disabled = false;
  button.classList.remove('disabled');
  node.each(function(d) {
      if (d.id in predictions) { // Verificar si el id del nodo está en predictions
          d3.select(this)
          .transition()
          .duration(1000)
          .attr("r", 12)
          .attr("fill", d => color(predictions[d.id]))
          .transition()
          .duration(1000)
          .attr("r", d => selectedNode && d.id === selectedNode ? 8 : 5);
      }
  });
}

function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
}

function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
}

function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
}



