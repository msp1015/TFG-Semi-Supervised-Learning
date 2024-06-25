function gettext(text) {
    return text; // Aquí gestionar la internacionalización
}

var matriz_confusion = [];
var mapa_clases = {};
var metricas_generales = {};
var metricas_clase = {};

function dibujaEstadisticas(datos) {
    matriz_confusion = datos.confusion_matrix;
    mapa_clases = JSON.parse(datos.mapa);
    metricas_generales = datos.metricas_generales;
    metricas_clase = datos.metricas_clase;

    const tabla = d3.select("#confusion-matrix");
    tabla.html("");

    const filaCabecera = tabla.append("tr");
    filaCabecera.append("th").text(gettext("Real/Predicción"));
    matriz_confusion[0].forEach((_, i) => {
        filaCabecera.append("th").text(gettext(mapa_clases[i]));
    });

    matriz_confusion.forEach((fila, i) => {
        const filaTabla = tabla.append("tr");
        filaTabla.append("th").text(gettext(mapa_clases[i]));
        fila.forEach((valor, j) => {
            filaTabla.append("td")
                .attr("id", `cell-${i}-${j}`)
                .text(valor);
        });
    });

    crearDropdown();
}

function crearDropdown() {
    const container = d3.select("#metrics-container");
    container.html("");

    const dropdownContainer = container.append("div")
        .attr("class", "dropdown-container");

    const select = dropdownContainer.append("select")
        .attr("id", "class-selector")
        .attr("class", "styled-dropdown")
        .on("change", actualizarGrafico);

    select.append("option")
        .attr("value", "general")
        .text(gettext("Vista General"));

    Object.keys(mapa_clases).forEach(key => {
        select.append("option")
            .attr("value", mapa_clases[key])
            .text(mapa_clases[key]);
    });

    actualizarGrafico(); // Mostrar gráfico inicial
}

function actualizarGrafico() {
    const seleccion = d3.select("#class-selector").node().value;
    const container = d3.select("#metrics-container");
    container.select("#chart").remove();
    container.select("#legend").remove();
    container.select("#metrics").remove();

    if (seleccion === "general") {
        resetearColores();
        mostrarMetricasGenerales(container);
        return;
    }

    const metricas = metricas_clase[seleccion];
    const data = [
        { label: "TP", value: metricas.TP },
        { label: "FP", value: metricas.FP },
        { label: "FN", value: metricas.FN },
        { label: "TN", value: metricas.TN }
    ];

    const width = 450;
    const height = 450;
    const margin = 40;
    const radius = Math.min(width, height) / 2 - margin;

    const svg = container.append("svg")
        .attr("id", "chart")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${width / 2}, ${height / 2})`);

    const color = d3.scaleOrdinal()
        .domain(data.map(d => d.label))
        .range(d3.schemeCategory10);

    const pie = d3.pie()
        .value(d => d.value);

    const data_ready = pie(data);

    const arc = d3.arc()
        .innerRadius(0)
        .outerRadius(radius);

    svg.selectAll('whatever')
        .data(data_ready)
        .join('path')
        .attr('d', arc)
        .attr('fill', d => color(d.data.label))
        .attr("stroke", "white")
        .style("stroke-width", "2px")
        .style("opacity", 0.7);

    svg.selectAll('whatever')
        .data(data_ready)
        .join('text')
        .text(d => d.data.value > 0 ? d.data.label + ":" + d.data.value : "")
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
                    .text(d.data.label + ":" + d.data.value);
                label.remove();
            }
        });

    const legend = container.append("div").attr("id", "legend");
    data.forEach(d => {
        const legendItem = legend.append("div").style("display", "flex").style("align-items", "center");
        legendItem.append("div")
            .style("width", "15px")
            .style("height", "15px")
            .style("background-color", color(d.label))
            .style("margin-right", "10px");
        legendItem.append("span").text(`${d.label}: ${d.desc}`);
    });

    colorearCeldas(seleccion, color);

    mostrarMetricasClase(container, metricas);
}

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

function mostrarMetricasGenerales(container) {
    const metricsDiv = container.append("div")
        .attr("id", "metrics")
        .attr("class", "metrics-summary");

    metricsDiv.append("h4").text(gettext("Métricas Generales"));
    Object.keys(metricas_generales).forEach(key => {
        metricsDiv.append("p").text(`${gettext(key)}: ${metricas_generales[key].toFixed(4)}`);
    });
}

function mostrarMetricasClase(container, metricas) {
    const metricsDiv = container.append("div")
        .attr("id", "metrics")
        .attr("class", "metrics-summary");

    metricsDiv.append("h4").text(gettext("Métricas de Clase"));
    Object.keys(metricas).forEach(key => {
        if (key !== "TP" && key !== "FP" && key !== "FN" && key !== "TN") {
            metricsDiv.append("p").text(`${gettext(key)}: ${metricas[key].toFixed(4)}`);
        }
    });
}

function resetearColores() {
    d3.selectAll("td").style("background-color", "");
}