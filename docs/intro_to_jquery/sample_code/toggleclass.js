$('#testbt').click(function() {

    // "test"라는 id를 가진 element에 클래스값(red)을 추가/삭제
    $('#test').toggleClass('red');
 })


// 초기상태 : 
// <p id='test' class='red'>Class name red!</p>

// 버튼클릭 : class 값이 삭제
// <p id='test'>Class name red!</p>

// 다시 버튼클릭 : class 값이 추가
// <p id='test' class='red'>Class name red!</p>

