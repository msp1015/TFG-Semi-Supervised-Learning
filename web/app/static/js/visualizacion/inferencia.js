
var matriz_confusion = [];
var mapa_clases = {};
var metricas_generales = {};
var metricas_clase = {};

/**
 * Gestiona la visualización de las estadísticas de inferencia.
 * Dibuja la matriz de confusión y las métricas generales y por clase.
 * 
 * @param {Object} datos 
 */
function dibujaEstadisticas(datos) {
    matriz_confusion = datos.confusion_matrix;
    mapa_clases = JSON.parse(datos.mapa);
    metricas_generales = datos.metricas_generales;
    metricas_clase = datos.metricas_clase;

    const tabla = d3.select("#confusion-matrix");
    tabla.html("");

    const filaCabecera = tabla.append("tr");
    filaCabecera.append("th").text(traducir("Real") + "\\" + 
    traducir("Prediction"));
    matriz_confusion[0].forEach((_, i) => {
        filaCabecera.append("th").text(mapa_clases[i]);
    });

    matriz_confusion.forEach((fila, i) => {
        const filaTabla = tabla.append("tr");
        filaTabla.append("th").text(mapa_clases[i]);
        fila.forEach((valor, j) => {
            filaTabla.append("td")
                .attr("id", `cell-${i}-${j}`)
                .text(valor);
        });
    });

    crearDropdown();
}

/**
 * Crea un dropdown para seleccionar la clase de la que se quieren ver las métricas.
 */
function crearDropdown() {
    const selector = d3.select("#selector");
    selector.html("");

    const dropdownContainer = selector.append("div")
        .attr("class", "dropdown-container");

    const select = dropdownContainer.append("select")
        .attr("id", "class-selector")
        .attr("class", "styled-dropdown")
        .on("change", actualizarGrafico);

    select.append("option")
        .attr("value", "general")
        .text(traducir("Average results"));

    Object.keys(mapa_clases).forEach(key => {
        select.append("option")
            .attr("value", mapa_clases[key])
            .text(mapa_clases[key]);
    });

    actualizarGrafico(); // Mostrar gráfico inicial
}

/**
 * Actualiza el gráfico de tarta con las métricas de la clase seleccionada.
 * Si se selecciona "general", se muestran las métricas generales.
 * Al seleccionar una clase, se resaltan las celdas de la matriz de confusión
 * correspondientes a esa clase.
 */
function actualizarGrafico() {
    const seleccion = d3.select("#class-selector").node().value;
    const container = d3.select("#metricas");
    container.select("#chart").remove();
    container.select("#resultados").remove();

    if (seleccion === "general") {
        resetearColores();
        mostrarMetricasGenerales(container);
        return;
    }

    const metricas = metricas_clase[seleccion];
    const data = [
        { etiq: "TP", valor: metricas.TP },
        { etiq: "FP", valor: metricas.FP },
        { etiq: "FN", valor: metricas.FN },
        { etiq: "TN", valor: metricas.TN }
    ];

    const width = 350;
    const height = 350;
    const margin = 40;
    const radius = Math.min(width, height) / 2 - margin;

    const containerTarta = container.append("div")
    .attr("id", "chart")
    .attr("class", "d-flex flex-column align-items-center");

    const svg = containerTarta.append("svg")
        .attr("id", "piechart")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${width / 2}, ${height / 2})`);

    const color = d3.scaleOrdinal()
        .domain(data.map(d => d.etiq))
        .range(d3.schemeCategory10);

    const pie = d3.pie()
        .value(d => d.valor);

    const data_ready = pie(data);

    const arc = d3.arc()
        .innerRadius(0)
        .outerRadius(radius);

    svg.selectAll('whatever')
        .data(data_ready)
        .join('path')
        .attr('d', arc)
        .attr('fill', d => color(d.data.etiq))
        .attr("stroke", "white")
        .style("stroke-width", "2px")
        .style("opacity", 0.7);

    svg.selectAll('whatever')
        .data(data_ready)
        .join('text')
        .text(d => d.data.valor > 0 ? d.data.etiq + ":" + d.data.valor : "")
        .attr("transform", d => `translate(${arc.centroid(d)})`)
        .style("text-anchor", "middle")
        .style("font-size", 15)
        .each(function(d) {
            const label = d3.select(this);
            if (d.endAngle - d.startAngle < 0.2) {
                const [lineX, lineY] = arc.centroid(d);
                const outerArc = d3.arc()
                    .innerRadius(radius)
                    .outerRadius(radius);

                const [outerX, outerY] = outerArc.centroid(d);
                label.attr("transform", `translate(${outerX}, ${outerY})`);
                
                let offset = 0;
                svg.selectAll('text')
                    .each(function() {
                        const textBBox = this.getBBox();
                        if (Math.abs(textBBox.y - (outerY + (outerY - lineY) * 0.3)) < 15) {
                            offset = 15;
                        }
                    });
                svg.append("line")
                    .attr("x1", outerX)
                    .attr("y1", outerY)
                    .attr("x2", outerX + (outerX - lineX) * 0.3)
                    .attr("y2", outerY + (outerY - lineY) * 0.3 + offset)
                    .attr("stroke", "black");
                svg.append("text")
                    .attr("x", outerX + (outerX - lineX) * 0.3)
                    .attr("y", outerY + (outerY - lineY) * 0.3 + offset)
                    .attr("dy", "0.35em")
                    .style("text-anchor", outerX > 0 ? "start" : "end")
                    .text(d.data.etiq + ":" + d.data.valor);
                label.remove();
            }
        });
    const containerPie = container.select("#chart");
    const leyenda = containerPie.append("div")
    .attr("id", "leyenda")
    .style("display", "grid") // Usar display grid para el contenedor de la leyenda
    .style("grid-template-columns", "repeat(2, 1fr)") // Crear dos columnas
    .style("gap", "10px"); // Espacio entre los elementos de la cuadrícula

    data.forEach(d => {
        const itemLeyenda = leyenda.append("div")
            .style("display", "flex") // Corregir "d-flex" a "flex"
            .style("align-items", "center"); // Alinear ítems al centro
        itemLeyenda.append("div")
            .style("width", "15px")
            .style("height", "15px")
            .style("background-color", color(d.etiq))
            .style("margin-right", "10px");
        itemLeyenda.append("span").text(`${traducir(d.etiq)}`);
    });

    colorearCeldas(seleccion, color);
    mostrarMetricasClase(container, metricas);
}

/**
 * Se encarga de colorear las celdas de la matriz de confusión según la clase seleccionada.
 * 
 * @param {string} clase 
 * @param {string} color 
 */
function colorearCeldas(clase, color) {
    const index = Object.keys(mapa_clases).find(key => mapa_clases[key] === clase);

    if (index !== undefined) {
        d3.selectAll("td").style("background-color", "");

        const tp = `cell-${index}-${index}`;
        d3.select(`#${tp}`).style("background-color", color("TP"));

        matriz_confusion.forEach((fila, i) => {
            fila.forEach((_, j) => {
                if (i === parseInt(index) && j !== parseInt(index)) {
                    d3.select(`#cell-${i}-${j}`).style("background-color", color("FN"));
                } else if (i !== parseInt(index) && j === parseInt(index)) {
                    d3.select(`#cell-${i}-${j}`).style("background-color", color("FP"));
                } else if (i !== parseInt(index) && j !== parseInt(index)) {
                    d3.select(`#cell-${i}-${j}`).style("background-color", color("TN"));
                }
            });
        });
    }
}

/**
 * Muestra las métricas generales cuando se selecciona "general" en el dropdown.
 * 
 * @param {Object} container 
 */
function mostrarMetricasGenerales(container) {
    const metricsDiv = container.append("div")
        .attr("id", "resultados")
        .attr("class", "m-2");

    metricsDiv.append("h4").text(traducir("Average metrics"));
    Object.keys(metricas_generales).forEach(key => {
        metricsDiv.append("p").text(`${traducir(key)}: ${metricas_generales[key].toFixed(4)}`);
    });
}

/**
 * Muestra las métricas de la clase seleccionada.
 * 
 * @param {Object} container 
 * @param {list} metricas 
 */
function mostrarMetricasClase(container, metricas) {
    const metricsDiv = container.append("div")
        .attr("id", "resultados")
        .attr("class", "m-2 p-2");

    metricsDiv.append("h4").text(traducir("Class metrics"));
    Object.keys(metricas).forEach(key => {
        if (key !== "TP" && key !== "FP" && key !== "FN" && key !== "TN") {
            metricsDiv.append("p").text(`${traducir(key)}: ${metricas[key].toFixed(4)}`);
        }
    });
}

/**
 * Restablece los colores de las celdas de la matriz de confusión.
 */
function resetearColores() {
    d3.selectAll("td").style("background-color", "");
}