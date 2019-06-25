
document.getElementById("submit").onclick = function()  {
    // var sentence = new FormData();
    // sentence.append("sentence", document.getElementById("sentence").value);
    //
    // alert(sentence);
    //
    // xmlHttp.open("POST", "http://localhost:8000/views", true);
    // xmlHttp.send(sentence);
    // document.getElementById("views").innerHTML = xmlHttp.responseText;

    fetchView();

    var result = document.getElementById("result")
    result.style.visibility = "visible"
}
function fetchView() {
    $.ajax({
        type: 'POST',
        dataType: 'JSON',
        contentType: 'application/x-www-form-urlencoded; charset=utf-8',
        url: getViewUrl.url + "/views",
        data: {"sentence": document.getElementById("sentence").value},
        success: function(views) {
            var viewlist = eval(views);
            var tbBody = "";
            $.each(viewlist, function(i, n) {

                var trColor;
                if (i % 2 == 0) {
                    trColor = "even";
                }
                else {
                    trColor = "odd";
                }
                tbBody += "<tr class='" + trColor + "'><td>" + n.person + "</td>" + "<td>" + n.verb + "</td>" + "<td>" + n.view + "</td></tr>";
                // $("#viewsBody").remove();
            });
            $("#viewsBody").html(tbBody);

        }

    });

    $("#loading_img").ajaxStart(function(){
        $(this).show();
    });

    $("#loading_img").ajaxStop(function(){
        $(this).hide();
    });
}

document.getElementById("clear").onclick = function() {
    document.getElementById("sentence").value  = "";
}
