/**
 *
 * Extrae el nombre del dataset eliminando
 * marcas temporales.
 * Ejemplo: iris.arff-32246473423
 * Resulta en: iris.arff
 *
 * Foro de ayuda: https://stackoverflow.com/questions/5202085/javascript-equivalent-of-pythons-rsplit
 *
 * @param file - nombre del fichero completo
 * @returns {*}
 */
function nombredataset(file) {
    return file.split('-').slice(0, -1).join('-');
}

let parametros_reales = {
    "en": {
        "target": "Target attribute",
        "cx": "X component",
        "cy": "Y component",
        "pca": "PCA",
        "stand": "Standardize",
        "p_unlabelled": "Unlabelled percentage",
        "p_test": "Test percentage",
        "nst": "N",
        "th": "Threshold",
        "n_iter": "Number of iterations",
        "p": "Positives",
        "nct": "Negatives",
        "u": "Number of initial data" },
    "es": {
        "target": "Atributo clase",
        "cx": "Componente X",
        "cy": "Componente Y",
        "pca": "PCA",
        "stand": "Estandarizar",
        "p_unlabelled": "Porcentaje de no etiquetados",
        "p_test": "Porcentaje de test",
        "nst": "N",
        "th": "Límite",
        "n_iter": "Número de iteraciones",
        "p": "Positivos",
        "nct": "Negativos",
        "u": "Número de iteraciones" }
};

const idiomas = {
    "en": {
        "decimal":        "",
        "emptyTable":     "No data available in table",
        "info":           "Showing _START_ to _END_ of _TOTAL_ entries",
        "infoEmpty":      "Showing 0 to 0 of 0 entries",
        "infoFiltered":   "(filtered from _MAX_ total entries)",
        "infoPostFix":    "",
        "thousands":      ",",
        "lengthMenu":     "Show _MENU_ entries",
        "loadingRecords": "Loading...",
        "processing":     "",
        "search":         "Search:",
        "zeroRecords":    "No matching records found",
        "paginate": {
            "first":      "First",
            "last":       "Last",
            "next":       "Next",
            "previous":   "Previous"
        }
    },
    "es": {
        "decimal": "",
        "emptyTable": "No hay información",
        "info": "Mostrando _START_ a _END_ de _TOTAL_ entradas",
        "infoEmpty": "Mostrando 0 to 0 of 0 entradas",
        "infoFiltered": "(Filtrado de _MAX_ total entradas)",
        "infoPostFix": "",
        "thousands": ",",
        "lengthMenu": "Mostrar _MENU_ entradas",
        "loadingRecords": "Cargando...",
        "processing": "Procesando...",
        "search": "Buscar:",
        "zeroRecords": "No hay datos en la tabla",
        "paginate": {
            "first": "Primero",
            "last": "Último",
            "next": "Siguiente",
            "previous": "Anterior"
        }
    }
}

const titulos = {'selftraining': 'Self-Training',
    'cotraining': 'Co-Training',
    'democraticcolearning': 'Democratic Co-Learning',
    'tritraining': 'Tri-Training',
    'coforest': 'Co-Forest',
    'graphs': 'GSSL'};

/**
 *
 * Realiza una petición GET para
 * obtener los datasets de los usuarios.
 *
 * Si id es null entonces se obtienen todos (se entiende
 * que no se quiere los de un usuario concreto).
 *
 * @param id - identificador del usuario
 * @returns {Promise<(*|string)[][]>}
 */
export const generateDatasetList = async (id=null) => {
    id = id==null ? '' : '/' + id;
    let response = await fetch('/datasets/obtener' + id);
    let data = await response.json();

    let datasets;

    if (Array.isArray(data)) {
        datasets = [];
        for (let dataset of data) {
            //["file","date", "user"]
            let aux = JSON.parse(dataset);

            datasets.push([nombredataset(aux[0]), aux[0], aux[1], aux[2], ""]);
        }
    } else {
        let aux = JSON.parse(data);
        datasets = [[nombredataset(aux[0]), aux[0], aux[1], aux[2], ""]];
    }
    return datasets;
}

/**
 *
 * Realiza una petición GET para
 * obtener las ejecuciones de los usuarios.
 *
 * Si id es null entonces se obtienen todos (se entiende
 * que no se quiere los de un usuario concreto).
 *
 * @param id - identificador del usuario
 * @returns {Promise<(*|string)[][]>}
 */
export const generateRunList = async (id=null) => {
    id = id==null ? '' : '/' + id;
    let response = await fetch('/historial/obtener' + id)
    let data = await response.json();

    let historial;
    if (Array.isArray(data)) {
        historial = [];
        for (let run of data) {
            //["id", "algorithm","filename","date","user", "cx", "cy", "jsonfile", "json_parameters"]
            let aux = JSON.parse(run);
            //Añadir una columna usuario en el <table>                             | //OCULTO
            historial.push([aux[1], nombredataset(aux[2]), aux[3], aux[10], aux[4], aux[0], aux[5], aux[6], aux[9]]);
        }
    } else {
        let aux = JSON.parse(data);
        historial = [[aux[1], nombredataset(aux[2]), aux[3], aux[10], aux[4], aux[0], aux[5], aux[6], aux[9]]];
    }
    return historial;
}

/**
 *
 * Realiza una petición GET para
 * obtener todos los usuarios.
 *
 * @returns {Promise<(*|string)[][]>}
 */
export const generateUserList = async () => {
    let response = await fetch('/usuarios/obtener');
    let data = await response.json();

    let usuarios;

    if (Array.isArray(data)) {
        usuarios = [];
        for (let usuario of data) {
            //["id","name","email","last_login"]
            let aux = JSON.parse(usuario);

            usuarios.push([aux[1], aux[2], aux[3], aux[0]]);
        }
    } else {
        let aux = JSON.parse(data);
        usuarios = [[aux[1], aux[2], aux[3], aux[0]]];
    }
    return usuarios;
}

/**
 *
 * Función común para la eliminación de los distintos
 * elementos con de ficheros: Datasets y Ejecuciones (Runs).
 *
 * @param ruta - ruta de eliminación
 * @param table - tabla sobre la que se actúa
 * @param row - fila concreta de la tabla
 * @param file - fichero a eliminar
 * @param id - identificador del fichero
 */
function fetch_eliminar(ruta, table, row, file, id) {
    fetch(ruta, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
        },
        body: JSON.stringify({
            "fichero": file,
            "id": id
        })
    }).then(function (response) {
        if (!response.ok){
            let error_modal = new bootstrap.Modal(document.getElementById('modal_error'));
            error_modal.show();
            response.json().then(mensaje => {
                document.getElementById('error_text').innerText = mensaje.error;
            })
                .catch(error => {console.log(error)});
        } else {
            row.remove().draw();

            if ('/datasets/eliminar' === ruta) { // Actualizar contadores
                let n_uploads = document.getElementById('n_uploads');
                if (n_uploads !== null) {
                    n_uploads.innerHTML = (parseInt(n_uploads.innerHTML) - 1).toString();
                }

                let recent_datasets = document.getElementById('recent_datasets');
                if (recent_datasets !== null) {
                    recent_datasets.innerHTML = (parseInt(recent_datasets.innerHTML) - 1).toString();
                }

            } else if ('/historial/eliminar' === ruta) { // Actualizar contadores
                let n_runs = document.getElementById('n_runs');
                if (n_runs !== null) {
                    n_runs.innerHTML = (parseInt(n_runs.innerHTML) - 1).toString();
                }

                let recent_runs = document.getElementById('recent_runs');
                if (recent_runs !== null) {
                    recent_runs.innerHTML = (parseInt(recent_runs.innerHTML) - 1).toString();
                }
            }

        }
    })
        .catch(error => console.log(error));
}

/**
 *
 * Función que asocia el evento de cambio de pestaña
 * en el panel de administrador para actualizar el tamaño
 * de la tabla y celdas (responsive).
 *
 * @param table - tabla sobre a la que asociar el evento
 */
function asociar_evento_resize_tab(table) {
    let lista_tabs = document.querySelectorAll('button[data-bs-toggle="tab"]');
    lista_tabs.forEach(function(tab){
        tab.addEventListener('shown.bs.tab', function (event) {
            table.columns.adjust().responsive.recalc();
        })
    });
}

// https://datatables.net/forums/discussion/54495/remove-table-row-from-a-button-inside-the-same-row-when-collapsed
// Transformado a vanilla
/**
 *
 * Crea la tabla de los datasets.
 * Sirve tanto para un usuario registrado como administrador.
 *
 * @param datasets - datos con los datasets
 * @param locale - idioma ('es' o 'en')
 * @param all_users - flag que indica si se quieren obtener los de todos los usuarios
 */
export function generateDatasetTable(datasets, locale, all_users) {

    let datasettable = document.querySelector('#datasettable');

    let table = new DataTable(datasettable, {
        "order": [[2, 'desc']],
        "responsive": true,
        "pageLength": 5,
        "language": idiomas[locale],
        "lengthMenu": [[5, 10, 20], [5, 10, 20]],
        "data": datasets,
        "columnDefs": [
            {"className": "align-middle", "targets": "_all"},
            {
                "targets": -1, // Columna acciones
                "className": "dt-body-center",
                "orderable": false,
                "render": function (data, type, row, meta) {

                    let acciones = '';
                    if (!all_users) {
                        acciones += '<button type="button" class="btn btn-warning run" data-file="' + row[1] + '">' +
                            '<div class="pe-none">' +
                            '<i class="bi bi-play-fill text-white"></i>' +
                            '</div>' +
                            '</button>'
                    }
                    acciones += '    <button class="btn btn-danger remove" data-file="' + row[1] + '">' +
                        '<div class="pe-none">' +
                        '<i class="bi bi-trash-fill text-white"></i>' +
                        '</div>' +
                        '</button>'

                    return acciones;
                }
            }, {
                "target": 1,
                "visible": false,
                "searchable": false,
            }, {
                "targets": -2, // Columna user
                "visible": all_users, // Si la tabla es para todos los datasets de todos los usuarios, esta columna será visible
                "searchable": all_users,
            }]
    });

    let id;

    if (all_users) {
        asociar_evento_resize_tab(table);
        id = -1;
    } else {
        id = document.querySelector('#user_id').value;
    }

    datasettable.addEventListener('click', function (event) {
        // Eliminar
        if (event.target.classList.contains('remove')) {
            let file = event.target.getAttribute('data-file');

            //https://datatables.net/forums/discussion/42918/child-row-how-to-select-parent-row
            //https://stackoverflow.com/questions/54477339/responsive-jquery-datatables-cannot-get-the-details-of-a-row
            let tr = event.target.closest('tr');
            if (tr.classList.contains('child')){
                tr = tr.previousSibling;
            }
            let row = table.row(tr);
            let row_data = row.data();

            let span_fichero = document.getElementById('nombre_fichero_modal');
            span_fichero.innerHTML = nombredataset(file);

            if (all_users) {
                span_fichero.innerHTML += ' (' + row_data[2] + ')'
            }

            let modal = new bootstrap.Modal(document.getElementById('modal_eliminar'));
            modal.show();

            let btn_eliminar = document.getElementById('btn_eliminar');
            btn_eliminar.onclick = function (e) {
                modal.hide()
                fetch_eliminar('/datasets/eliminar', table, row, file, id);
            }
            // Ejecutar
        } else if (event.target.classList.contains('run')) {
            let file = event.target.getAttribute('data-file');

            let modal = new bootstrap.Modal(document.getElementById('modal_ejecutar'));
            modal.show();

            let selftraining = document.getElementById('selftraining_link');
            selftraining.setAttribute('href', '/seleccionar/selftraining/' + file);

            let cotraining = document.getElementById('cotraining_link');
            cotraining.setAttribute('href', '/seleccionar/cotraining/' + file);

            let democraticcolearning = document.getElementById('democraticcolearning_link');
            democraticcolearning.setAttribute('href', '/seleccionar/democraticcolearning/' + file);

            let triraining = document.getElementById('tritraining_link');
            triraining.setAttribute('href', '/seleccionar/tritraining/' + file);

            let coforest = document.getElementById('coforest_link');
            coforest.setAttribute('href', '/seleccionar/coforest/' + file);

            let graphs = document.getElementById('graphs_link');
            graphs.setAttribute('href', '/seleccionar/graphs/' + file);

        }
    });
}

/**
 *
 * Crea la tabla de las ejecuciones.
 * Sirve tanto para un usuario registrado como administrador.
 *
 * @param historial - datos con las ejecuciones (un historial)
 * @param locale - idioma ('es' o 'en')
 * @param all_users - flag que indica si se quieren obtener los de todos los usuarios
 */
export function generateHistoryTable(historial, locale, all_users) {

    let historytable = document.querySelector('#historytable');

    let table = new DataTable(historytable, {
        "order": [[2, 'desc']],
        "responsive": true,
        "pageLength": 5,
        "language": idiomas[locale],
        "lengthMenu": [[5, 10, 20], [5, 10, 20]],
        "data": historial,
        "columnDefs": [
            {"className": "align-middle", "targets": "_all"},
            {
                "targets": 0,
                "render": function (data, type, row, meta) {
                    return titulos[row[0]]; // Sustituir nombre del algoritmo por su título
                }
            },
            {
                "targets": 3, // Columna acciones
                "className": "dt-body-center",
                "orderable": false,
                "render": function (data, type, row, meta) {
                    return '<button class="btn btn-success parameters">' +
                        '<div class="pe-none">' +
                        '<i class="bi bi-file-earmark-spreadsheet-fill"></i>' +
                        '</div>' +
                        '</button>';
                }
            },
            {
                "targets": -1, // Columna acciones
                "className": "dt-body-center",
                "orderable": false,
                "render": function (data, type, row, meta) {
                    let acciones = '';
                    if (!all_users) {
                        acciones += '<a type="button" class="btn btn-warning run" href="/visualizacion/' + row[0] +'/' + row[5] +'">' +
                            '<div class="pe-none">' +
                            '<i class="bi bi-arrow-clockwise text-white"></i>' +
                            '</div>' +
                            '</a>'
                    }
                    acciones += '    <button class="btn btn-danger remove" data-file="' + row[8] + '">' +
                        '<div class="pe-none">' +
                        '<i class="bi bi-trash-fill text-white"></i>' +
                        '</div>' +
                        '</button>'

                    return acciones;
                }
            }, {
                "targets": -2, // Columna user
                "visible": all_users, // Si la tabla es para todos los datasets de todos los usuarios, esta columna será visible
                "searchable": all_users,
            }]
    });

    let id;

    if (all_users) {
        asociar_evento_resize_tab(table);
        id = -1;
    } else {
        id = document.querySelector('#user_id').value;
    }

    historytable.addEventListener('click', function (event) {
        // Eliminar
        if (event.target.classList.contains('remove')) {
            let file = event.target.getAttribute('data-file');

            let tr = event.target.closest('tr');
            if (tr.classList.contains('child')){
                tr = tr.previousSibling;
            }
            let row = table.row(tr);
            let row_data = row.data();

            let span_fichero = document.getElementById('nombre_fichero_modal');
            span_fichero.innerHTML = titulos[row_data[0]] + ' - ' + row_data[1] + ' (' + row_data[2] + ')';

            let modal = new bootstrap.Modal(document.getElementById('modal_eliminar'));
            modal.show();

            let btn_eliminar = document.getElementById('btn_eliminar');
            btn_eliminar.onclick = function (e) {
                modal.hide();
                fetch_eliminar('/historial/eliminar', table, row, file, id);
            }
        } else if (event.target.classList.contains('parameters')){
            let tr = event.target.closest('tr');
            if (tr.classList.contains('child')){
                tr = tr.previousSibling;
            }
            let row = table.row(tr);
            let row_data = row.data();

            let json = JSON.parse(row_data[3]);
            let modal = new bootstrap.Modal(document.getElementById('modal_parametros'));
            modal.show();
            console.log(row_data)
            console.log(row_data[0])
            console.log(row_data[1])
            console.log(row_data[2])
            console.log(row_data[3])
            let readable_json = {}
            
            for (let key of Object.keys(json)) {
                console.log(key);
                let aux = key;
                if (key === "n") {
                    if (row_data[0] === "selftraining") {

                        aux += "st";
                    } else if (row_data[0] === "cotraining") {
                        aux += "ct";
                    }
                }
                if (aux in parametros_reales[locale]) {
                    readable_json[parametros_reales[locale][aux]] = json[key]
                } else {
                    readable_json[aux] = json[key];
                }
            }

            document.getElementById("json_parameters").innerHTML = JSON.stringify(readable_json, null, "  ");



        }
    });
}

/**
 *
 * Crea la tabla de los usuarios.
 * Sirve para el administrador.
 *
 * @param usuarios datos con las ejecuciones (un historial)
 * @param locale idioma ('es' o 'en')
 */
export function generateUserTable(usuarios, locale) {

    let usertable = document.querySelector('#usertable');

    let table = new DataTable(usertable, {
        "order": [[2, 'desc']],
        "responsive": true,
        "pageLength": 5,
        "language": idiomas[locale],
        "lengthMenu": [[5, 10, 20], [5, 10, 20]],
        "data": usuarios,
        "columnDefs": [
            {"className": "align-middle", "targets": "_all"},
            {
                "targets": -1, // Columna acciones
                "className": "dt-body-center",
                "orderable": false,
                "render": function (data, type, row, meta) {
                    return '<a type="button" class="btn btn-success edit" href="/admin/usuario/editar/' + row[3] + '">' +
                        '<div class="pe-none">' +
                        '<i class="bi bi-pencil-fill text-white"></i>' +
                        '</div>' +
                        '<a>'
                        +'  <button class="btn btn-danger remove" data-user="' + row[3] + '">' +
                        '<div class="pe-none">' +
                        '<i class="bi bi-trash-fill text-white"></i>' +
                        '</div>' +
                        '</button>';
                }
            }]
    });

    asociar_evento_resize_tab(table);

    usertable.addEventListener('click', function (event) {
        // Eliminar
        if (event.target.classList.contains('remove')) {
            let user = event.target.getAttribute('data-user');

            let tr = event.target.closest('tr');
            if (tr.classList.contains('child')){
                tr = tr.previousSibling;
            }
            let row = table.row(tr);
            let row_data = row.data();

            let span_usuario = document.getElementById('nombre_fichero_modal');
            span_usuario.innerText = row_data[0] + ' (' + row_data[1] + ')' ;

            let modal = new bootstrap.Modal(document.getElementById('modal_eliminar'));
            modal.show();

            let btn_eliminar = document.getElementById('btn_eliminar');
            btn_eliminar.onclick = function (e) {
                modal.hide();
                fetch('/usuarios/eliminar', {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
                    },
                    body: JSON.stringify({
                        "user_id": user
                    })
                }).then(function (response) {
                    if (!response.ok){
                        let error_modal = new bootstrap.Modal(document.getElementById('modal_error'));
                        error_modal.show();
                        response.json().then(mensaje => {
                            document.getElementById('error_text').innerText = mensaje.error;
                        })
                            .catch(error => {console.log(error)});
                    } else {
                        row.remove().draw();
                        location.reload();
                    }
                })
                    .catch(error => console.log(error));

            }
        }
    });
}
