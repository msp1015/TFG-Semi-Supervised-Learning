document.addEventListener('DOMContentLoaded', function () {
    ['dragleave', 'drop', 'dragenter', 'dragover'].forEach(function (evento) {
        document.addEventListener(evento, function (e) {
            e.preventDefault();
        }, false);
    });

    let area = document.getElementById('soltar');
    area.param = 'soltar';
    
    let progreso = document.getElementById('progreso');
    let porcentaje_progreso = document.getElementById('porcentaje_progreso');
    let nombre_fichero = document.getElementById('nombre_fichero');
    let titulo_fichero = document.getElementById('titulo_fichero');
    let boton = document.getElementById('archivo');
    let tableWarning = document.getElementById('table_warning');
    let tableContainer = document.getElementById('table_container');
    let dataContainer = document.getElementById('container_data');
    boton.param = 'boton';

    area.addEventListener('drop', subir, false)
    boton.addEventListener('change', subir)


    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(function (el) {
        new bootstrap.Tooltip(el);
    });

    let btn_condiciones = document.getElementById('btn_condiciones');
    btn_condiciones.onclick = function (e) {
        let modal = new bootstrap.Modal(document.getElementById('modal_condiciones'));
        modal.show();
    }

    /**
     *
     * Gestiona la subida del fichero realizando una petición post
     * sobre una ruta de la aplicación.
     * Gracias al evento del proceso permite determinar el porcentaje de subida.
     *
     * @param e - evento
     * @returns {boolean}
     */
    function subir(e) {
        e.preventDefault();
        let archivo;

        if (e.currentTarget.param === 'soltar') {
            archivo = e.dataTransfer.files; // arrastar y soltar
        } else {
            archivo = e.target.files; // botón de subida
        }
        if (archivo.length >= 2) {
            return false;
        }

        nombre_fichero.textContent = archivo[0].name;
        let xhr = new XMLHttpRequest();
        xhr.open('post', '/subida', true);
        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4){
                if(xhr.status === 200) {
                    let button = document.querySelector("#fichero-previo");
                    if (button == null) {
                        document.getElementById("config_btn").disabled = false;
                    } else {
                        button.style.display = "none";
                    }    
                    titulo_fichero.innerText = archivo[0].name;
                    tableContainer.style.display = 'block';
                    dataContainer.style.display = 'block';
                    tableWarning.style.display = 'none';
                    actualizarTabla();
                }
            }
        };

        xhr.upload.onprogress = function (evento) {
            if (evento.lengthComputable) {
                let porcentaje = Math.floor(evento.loaded / evento.total * 100);
                progreso.style.width = porcentaje.toString() + "%";
                progreso.setAttribute("aria-valuenow", porcentaje.toString());
                porcentaje_progreso.textContent = porcentaje.toString() + "%";
            }
        };

        xhr.setRequestHeader('X-CSRFToken', document.querySelector('meta[name="csrf-token"]').getAttribute('content'));
        let params = new FormData();
        params.append('archivo', archivo[0])
        xhr.send(params);
    }

    // Llama a actualizarTabla al cargar la página para manejar el caso en que ya exista un fichero en sesión
    actualizarTabla();
});

function actualizarTabla() {

    const idiomasDataTables = {
        'es': '//cdn.datatables.net/plug-ins/1.10.21/i18n/Spanish.json',
        'en': '//cdn.datatables.net/plug-ins/1.10.21/i18n/English.json'
    };
    const urlIdioma = idiomasDataTables[locale] || idiomasDataTables['en'];

    fetch('/obtenerDatosTabla')
        .then(response => {
            if (!response.ok) {
                throw new Error('Error al obtener los datos de la tabla');
            }
            return response.json();
        })
        .then(data => {
            if (data.status && data.status === "ERROR") {
                // Si el cuerpo de la respuesta contiene un estado de error, lanza una excepción con el mensaje de error
                throw new Error(data.error);
            }
            if ($.fn.DataTable.isDataTable('#csvTable')) {
                // Destruir la instancia actual de DataTable para resetear completamente la tabla
                $('#csvTable').DataTable().destroy();
                
                // Limpiar el contenido HTML de la tabla para eliminar las columnas antiguas
                $('#csvTable').empty();
            }
            
            // Inicializar DataTable con los nuevos datos y configuraciones de columnas
            $('#csvTable').DataTable({
                data: data.data,
                columns: data.columns,
                language: {
                    url: urlIdioma
                },
                responsive: true,
                paging: true,
                pageLength: 25,
                lengthChange: true,
                scrollY: '50vh',
                scrollX: true
            });
        }).catch(error => {
            console.error('Error actualizando la tabla:', error);
            let tableWarning = document.getElementById('table_warning');
            let csvContainer = document.getElementById('container_data');
            csvContainer.style.display = 'none';
            tableWarning.style.display = 'block';
        });
}

function establecerFicheroPrueba(fichero) {
    fetch('/establecer_prueba', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': document.querySelector('meta[name="csrf-token"]').getAttribute('content')
        },
        body: JSON.stringify({ fichero: fichero })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error(data.error);
            alert(data.error);
        } else {
            let texto = document.querySelector("#fichero-previo");
            if (texto == null) {
                document.getElementById("config_btn").disabled = false;
            } else {
                texto.style.display = "none";
            }
            let titulo_fichero = document.getElementById('titulo_fichero');
            titulo_fichero.innerText = fichero + '.arff';
            let tableWarning = document.getElementById('table_warning');
            let csvContainer = document.getElementById('container_data');
            csvContainer.style.display = 'block';
            tableWarning.style.display = 'none';
            let nombre_fichero = document.getElementById('nombre_fichero');
            nombre_fichero.textContent = "";
            let tableContainer = document.getElementById('table_container');
            tableContainer.style.display = 'block';
            actualizarTabla();
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}