/**
 *
 * Se encarga de generar el gráfico de las estadísticas específicas.
 * La zona del gráfico se divide en tres:
 *  - Selector de la estadística
 *  - Gráfico SVG
 *  - Checboxes para seleccionar los clasificadores
 *
 * @param specific_stats - datos con las estadísticas específicas
 */
function generarespecificas(specific_stats) {
    const id_div_especificas = "estadisticas_especificas";
    let div_especificas = document.querySelector("#" + id_div_especificas);

    let clasificadores = Object.keys(specific_stats);
    let datos_json = JSON.parse(specific_stats[clasificadores[0]]);
    let stats = Object.keys(datos_json);

    // Tres partes del gráfico
    let id_selector_stat = "selector_stat";
    let id_div_estadisticas = "grafico_stat";
    let id_div_checboxes = "checkboxes_especifico_grafico_stat";


    // PARTE 1: Crear selector para las estadísticas
    let select = document.createElement("select");
    select.setAttribute("id", id_selector_stat);
    select.setAttribute("class", "form-select");
    select.style.width = "25%";
    select.style.margin = "auto";

    for (let stat of stats) {
        let option = document.createElement("option");
        option.value = stat;
        option.text = stat;
        select.appendChild(option);
    }

    // El selector hará que en el SVG se modifique la estadística mostrada
    select.addEventListener('change', (event) => {
        seleccionarstat(id_div_estadisticas, event.target.value, stats);
    });

    div_especificas.appendChild(select);

    // PARTE 2: Gráfico SVG
    let div_grafico = document.createElement("div");
    div_grafico.setAttribute("id", id_div_estadisticas);
    div_especificas.appendChild(div_grafico);
    // Con null se indica que no genere las líneas de estadísticas, se delega a esta misma (actual) función
    generargraficoestadistico(id_div_estadisticas, null, clasificadores);

    let color = d3.scaleOrdinal()
        .domain(clasificadores)
        .range(d3.schemeCategory10);

    for (let clf of clasificadores) {
        let datos_json = JSON.parse(specific_stats[clf]);

        for (let stat of stats) {
            anadirestadistica(id_div_estadisticas, datos_json, stat, color, clf);
        }
    }


    // PARTE 3: Checkboxes de los clasificadores
    let div_checkboxes = document.createElement("div");
    div_checkboxes.classList.add("mt-3");
    div_checkboxes.setAttribute("id", id_div_checboxes);
    div_especificas.appendChild(div_checkboxes);
    generarcheckboxes_clasificadores(id_div_checboxes, id_div_estadisticas, clasificadores);


    // Para que al cargar la página, muestre la primera estadística
    seleccionarstat(id_div_estadisticas, stats[0], stats);

}

/**
 *
 * Controla la visibilidad de las líneas de estadísticas.
 * Al seleccionar una mediante el selector, hará visible la estadística
 * seleccionada, y el resto serán ocultadas.
 *
 * @param id_div_estadisticas - id del gráfico SVG
 * @param stat_seleccionada - nombre de la estadística seleccionada (del selector)
 * @param lista_stats - lista de todos los nombres de las estadísticas (utilizada para comparar)
 */
function seleccionarstat(id_div_estadisticas, stat_seleccionada, lista_stats) {
    let statsvg = d3.select("#" + id_div_estadisticas).select("svg").select("g");

    for (let stat of lista_stats) {
        let pts = statsvg.selectAll('circle[stat='+ stat +']');
        let lineas = statsvg.selectAll('line[stat='+ stat +']');
        if (stat_seleccionada === stat){
            // Aparte de ser la estadística seleccionada se tiene que cumplir que:
            // La estadística de los puntos y líneas deben ser solo las de la estadística seleccionada (en el select)
            // El clasificador de los puntos y líneas debe ser alguno de los seleccionados en los checkboxes
            pts.filter(function(d) {
                return d[0] <= cont && comprobarvisibilidad(d3.select(this).attr("clf"),stat_seleccionada);
            })
                .style("visibility", "visible")

            lineas.filter(function (d) {
                return d <= cont && comprobarvisibilidad(d3.select(this).attr("clf"),stat_seleccionada);
            })
                .style("visibility", "visible")
        }else{
            pts.style("visibility", "hidden");
            lineas.style("visibility", "hidden");
        }
    }

}
/**
 *
 * Genera los checkboxes de cada clasificador
 *
 * @param id_div_objetivo - id del <div> donde se incluirán los checkboxes
 * @param id_div_estadisticas - id del <div> del gráfico SVG
 * @param clasificadores - lista de los nombres de los clasificadores
 */
function generarcheckboxes_clasificadores(id_div_objetivo, id_div_estadisticas, clasificadores) {
    let div_objetivo = document.querySelector("#" + id_div_objetivo);

    // Crear checkbox para seleccionar todos
    let selectAllCheckbox = document.createElement("input");
    selectAllCheckbox.setAttribute("id", "chk_select_all");
    selectAllCheckbox.setAttribute("type", "checkbox");
    selectAllCheckbox.setAttribute("class", "form-check-input");
    selectAllCheckbox.addEventListener("change", function() {
        let checkboxes = div_objetivo.querySelectorAll("input[type='checkbox']");
        checkboxes.forEach(checkbox => {
            if (checkbox.id !== "chk_select_all" && checkbox.id !== "chk_deselect_all") {
                checkbox.checked = true;
                habilitar_clasificador(id_div_estadisticas, true, checkbox.id.replace("chk_", ""));
            }
        });
        if (this.checked) {
            deselectAllCheckbox.checked = false;
        }
        actualizarSeleccionTodosNinguno();
    });
    div_objetivo.appendChild(selectAllCheckbox);
    div_objetivo.appendChild(document.createTextNode(traducir('All')));

    // Crear checkbox para deseleccionar todos
    let deselectAllCheckbox = document.createElement("input");
    deselectAllCheckbox.setAttribute("id", "chk_deselect_all");
    deselectAllCheckbox.setAttribute("type", "checkbox");
    deselectAllCheckbox.setAttribute("class", "form-check-input");
    deselectAllCheckbox.addEventListener("change", function() {
        let checkboxes = div_objetivo.querySelectorAll("input[type='checkbox']");
        checkboxes.forEach(checkbox => {
            if (checkbox.id !== "chk_select_all" && checkbox.id !== "chk_deselect_all") {
                checkbox.checked = false;
                habilitar_clasificador(id_div_estadisticas, false, checkbox.id.replace("chk_", ""));
            }
        });
        if (this.checked) {
            selectAllCheckbox.checked = false;
        }
        actualizarSeleccionTodosNinguno();
    });
    div_objetivo.appendChild(deselectAllCheckbox);
    div_objetivo.appendChild(document.createTextNode(traducir('None')));

    for (let index in clasificadores) {
        let clasificador = clasificadores[index];

        let input = document.createElement("input");
        input.setAttribute("id", "chk_" + clasificador.replace("(", "").replace(")", ""));
        input.setAttribute("type", "checkbox");
        input.setAttribute("class", "form-check-input");
        input.checked = true;

        input.addEventListener("change", function() {
            habilitar_clasificador(id_div_estadisticas, this.checked, clasificador);
            actualizarSeleccionTodosNinguno();
        });

        let label = document.createElement("label");
        label.setAttribute("for", "chk_" + clasificador.replace("(", "").replace(")", ""));
        label.textContent = clasificador;

        div_objetivo.appendChild(input);
        div_objetivo.appendChild(label);
        div_objetivo.append(" ");
    }

    /**
     * Actualiza el estado de "marcar todos" y "desmarcar todos"
     */
    function actualizarSeleccionTodosNinguno() {
        let checkboxes = div_objetivo.querySelectorAll("input[type='checkbox']");
        let individualCheckboxes = Array.from(checkboxes).filter(chk => chk.id !== "chk_select_all" && chk.id !== "chk_deselect_all");

        let allChecked = individualCheckboxes.every(chk => chk.checked);
        let noneChecked = individualCheckboxes.every(chk => !chk.checked);

        selectAllCheckbox.checked = allChecked;
        deselectAllCheckbox.checked = noneChecked;

        selectAllCheckbox.disabled = allChecked;
        deselectAllCheckbox.disabled = noneChecked;
    }

    // Inicializar el estado de los botones "All" y "None"
    actualizarSeleccionTodosNinguno();
}

/**
 *
 * Deshabilita/habilita los puntos y líneas dependiendo del
 * clasificador seleccionado en los checboxes.
 *
 * Es una función que gestiona el evento "clic" en un checkbox.
 *
 * @param id_div_objetivo - id del <div> del gráfico SVG
 * @param checked - valor booleano que indica que si el checboxes está activo o no
 * @param clasificador - nombre del clasificador
 */
function habilitar_clasificador(id_div_objetivo, checked, clasificador) {
    let nombre_clasificador = clasificador.replace("(", "").replace(")", "");
    let statsvg = d3.select("#" + id_div_objetivo).select("svg").select("g");
    let pts = statsvg.selectAll('circle[clf=' + nombre_clasificador + ']');
    let lineas = statsvg.selectAll('line[clf=' + nombre_clasificador + ']');

    if (checked) {
        pts.filter(function(d) {
            return d[0] <= cont && comprobarvisibilidad(nombre_clasificador, d3.select(this).attr("stat"));
        }).style("visibility", "visible");

        lineas.filter(function(d) {
            return d <= cont && comprobarvisibilidad(nombre_clasificador, d3.select(this).attr("stat"));
        }).style("visibility", "visible");
    } else {
        pts.style("visibility", "hidden");
        lineas.style("visibility", "hidden");
    }
}


/**
 *
 * Función general para crear el gráfico SVG, sin puntos
 * ni líneas, solo la declaración del mismo.
 * También genera la leyenda.
 *
 * @param id_div_objetivo - id del <div> donde se alojará el SVG
 * @param datos_stats - datos de todas las estadísticas
 * @param dominio - el dominio de la leyenda (lo nombres a mostrar)
 */
function generargraficoestadistico(id_div_objetivo, datos_stats, dominio) {

    let margin = {top: 10, right: 10, bottom: 50, left: 50},
        width = 650 - margin.left - margin.right,
        height = 325 - margin.top - margin.bottom;

    let statsvg = d3.select("#" + id_div_objetivo)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .style("margin", "auto");

    let statx = d3.scaleLinear()
        .domain([0, maxit])
        .range([0, width]);

    let staty = d3.scaleLinear()
        .domain([0, 1])
        .range([height, 0]);

    function numero_ticks() {
        let resto = Math.ceil(maxit / 35);
        if (resto <= 1 ) {
            return maxit;
        } else {
            return Math.ceil(maxit/resto);
        }
    }

    statsvg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(statx).ticks(numero_ticks()));

    statsvg.append("g")
        .call(d3.axisLeft(staty));

    //Etiqueta eje X
    statsvg.append("text")
        .attr("class", "cx")
        .attr("text-anchor", "middle")
        .attr("x", width / 2)
        .attr("y", height + margin.bottom)
        .text(traducir('Iteration'));

    //Etiqueta eje Y
    statsvg.append("text")
        .attr("class", "cy")
        .attr("text-anchor", "middle")
        .attr("y", -margin.left)
        .attr("x", -height / 2)
        .attr("dy", "1em")
        .attr("transform", "rotate(-90)")
        .text(traducir('Measure'));

    let color = d3.scaleOrdinal()
        .domain(dominio)
        .range(d3.schemeCategory10);

    let leyenda;

    if (datos_stats !== null) {
        leyenda = document.getElementById("leyenda_estadisticas_generales");
        leyenda.innerHTML = "";
    }else{
        leyenda = document.getElementById("leyenda_estadisticas_especificas");
        leyenda.innerHTML = "";
    }
    leyenda.style.maxHeight = "300px"; // Ajusta este valor según sea necesario
    leyenda.style.overflowY = "auto";
    for (const d of dominio) {
        let span = document.createElement("span");
        span.classList.add("text-left");
        span.classList.add("text-break");
        span.style.color = color(d);
        span.innerHTML = d;
        leyenda.appendChild(span);
    }

    // Con el gráfico creado se dibuja cada estadística
    // vinculando los eventos
    if (datos_stats !== null) {
        for (let index in dominio) {
            anadirestadistica(id_div_objetivo, datos_stats, dominio[index], color);
        }
    }

}

/**
 *
 * Añade los puntos y las líneas de las estadísticas.
 * Se utiliza tanto para las estadísticas generales como para las específicas.
 *
 * @param id_div - id del <div> del SVG de las estadísticas
 * @param datos_stats - datos de las estadísticas
 * @param stat - nombre de la estadística a añadir
 * @param color - rango de colores
 * @param clf - indica si las líneas y puntos corresponde a un clasificador concreto o no
 */
function anadirestadistica(id_div, datos_stats, stat, color, clf="no") {
    let margin = {top: 10, right: 10, bottom: 50, left: 50},
        width = 650 - margin.left - margin.right,
        height = 325 - margin.top - margin.bottom;

    let lista = crearListaStat(datos_stats, stat);

    let statsvg = d3.select("#" + id_div).select("svg").select("g");

    let statx = d3.scaleLinear()
        .domain([0, maxit])
        .range([0, width]);

    let staty = d3.scaleLinear()
        .domain([0, 1])
        .range([height, 0]);

    let nombre_clf = clf.replace("(","").replace(")","");

    let pts = statsvg.selectAll("dot")
        .data(lista)
        .enter()
        .append("circle")
        .attr("clf", nombre_clf)
        .attr("stat", stat)
        .attr("cx", function (d) {
            return statx(d[0]);
        })
        .attr("cy", function (d) {
            return staty(d[1]);
        })
        .attr("r", 5)
        .attr("fill", function () {
            if (clf === "no"){
                return color(stat);
            }else{
                return color(clf);
            }

        })
        .style("visibility", function (d) {
            if (d[0] <= cont) {
                return "visible";
            } else {
                return "hidden";
            }
        })

    for (let i = 0; i < lista.length - 1; i++) {
        statsvg.append("line")
            .data([i + 1])
            .attr("clf", nombre_clf)
            .attr("stat", stat)
            .attr("x1", statx(i))
            .attr("y1", staty(lista[i][1]))
            .attr("x2", statx(i + 1))
            .attr("y2", staty(lista[i + 1][1]))
            .attr("stroke", function () {
                if (clf === "no"){ // Estadísticas generales
                    return color(stat);
                }else{
                    return color(clf);
                }
            })
            .attr("stroke-width", 1.5)
            .style("visibility", function (d) {
                if (d < cont) {
                    return "visible";
                } else {
                    return "hidden";
                }
            })
    }

    let lineas = statsvg.selectAll('line[clf=' + nombre_clf + '][stat=' + stat + ']');

    document.addEventListener('next', next);
    document.addEventListener('prev', prev);

    /**
     * Muestra los puntos y líneas de la siguiente iteración
     */
    function next() {
        pts.filter(function (d) {
            return d[0] === cont && comprobarvisibilidad(nombre_clf, stat);
        })
            .style("opacity", 0)
            .transition()
            .duration(600)
            .style("opacity", 1)
            .style("visibility", "visible");

        lineas.filter(function (d) {
            return d === cont && comprobarvisibilidad(nombre_clf, stat);
        })
            .attr("stroke-width", 0)
            .transition()
            .duration(600)
            .attr("stroke-width", 1)
            .style("visibility", "visible");

    }

    /**
     * Oculta los puntos y líneas de iteraciones posteriores
     */
    function prev() {
        pts.filter(function (d) {
            return d[0] > cont;
        })
            .style("visibility", "hidden");

        lineas.filter(function (d) {
            return d > cont;
        })
            .style("visibility", "hidden");
    }
}

/**
 *
 * Crea un array de arrays con el valor de la estadística
 * por cada iteración
 *
 * @param stats - datos de las estadísticas
 * @param stat - nombre de la estadística
 * @returns {*[]} - array de arrays
 */
function crearListaStat(stats,stat){
    let lista = [];
    let aux = stats[stat];

    for (let i = 0; i <= maxit; i++) {
        lista.push([i,aux[i]]);
    }

    return lista;
}

/**
 *
 * Para las estadísticas específicas, comprueba que:
 * Si la estadística del selector es la que se comprueba
 * El checkbox del clasificador base está activado
 *
 * @param clasificador - nombre del clasificador
 * @param stat - nombre de la estadística
 * @returns {boolean} - true si se cumplen las dos condiciones, false en caso contrario
 */
function comprobarvisibilidad(clasificador, stat) {
    let check = document.getElementById("chk_" + clasificador);
    let select = document.getElementById("selector_stat");

    if (check === null){ //Estadísticas generales
        return true;
    }else{
        if (check.checked && select.value === stat){
            return true;
        }
    }
    return false;
}