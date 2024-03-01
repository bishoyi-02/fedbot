package com.chaquo.python.utils;

import androidx.annotation.NonNull;

import android.content.Context;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import com.chaquo.python.console.R;
import android.view.View;
import android.app.*;
import android.content.DialogInterface;
import android.graphics.*;
import android.os.*;
import android.text.*;
import android.text.style.*;
import android.util.Log;
import android.view.*;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.view.inputmethod.*;
import android.widget.*;
import androidx.annotation.*;
import androidx.appcompat.app.*;
import androidx.appcompat.app.AlertDialog;
import androidx.core.app.ActivityCompat;
import androidx.core.content.*;
import androidx.lifecycle.*;
import com.chaquo.python.Python;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.Socket;
import java.net.SocketException;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.util.Enumeration;

public abstract class ConsoleActivity extends AppCompatActivity
implements ViewTreeObserver.OnGlobalLayoutListener, ViewTreeObserver.OnScrollChangedListener {

    // Because tvOutput has freezesText enabled, letting it get too large can cause a
    // TransactionTooLargeException. The limit isn't in the saved state itself, but in the
    // Binder transaction which transfers it to the system server. So it doesn't happen if
    // you're rotating the screen, but it does happen when you press Back.
    //
    // The exception message shows the size of the failed transaction, so I can determine from
    // experiment that the limit is about 500 KB, and each character consumes 4 bytes.
    private final int MAX_SCROLLBACK_LEN = 100000;
    private String ip;

    private Socket client;
    private PrintWriter printwriter;
    private EditText textField;
    private Button button;
    private String message;
    private EditText etInput;
    private ScrollView svOutput;
    private TextView tvOutput;
    private int outputWidth = -1, outputHeight = -1;

    ImageButton sendSignalButton,uploadButton,downloadButton,trainButton,expandButton;

    public  boolean isStoragePermissionGranted() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED) {
                return true;
            } else {
                ActivityCompat.requestPermissions(this, new String[]{ android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
                return false;
            }
        }
        else { //permission is automatically granted on sdk<23 upon installation
            return true;
        }
    }
    public static String getLocalIpAddress() {
        try {
            for (Enumeration<NetworkInterface> en = NetworkInterface.getNetworkInterfaces(); en.hasMoreElements();) {
                NetworkInterface intf = en.nextElement();
                for (Enumeration<InetAddress> enumIpAddr = intf.getInetAddresses(); enumIpAddr.hasMoreElements();) {
                    InetAddress inetAddress = enumIpAddr.nextElement();
                    if (!inetAddress.isLoopbackAddress() && inetAddress instanceof Inet4Address) {
                        return inetAddress.getHostAddress();
                    }
                }
            }
        } catch (SocketException ex) {
            ex.printStackTrace();
        }
        return null;
    }
    public void toast(String text){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast toast = Toast.makeText(ConsoleActivity.this, text, Toast.LENGTH_SHORT);
                toast.show();
            }
        });

    }
    public void trainModel(View view) {
        Runnable runnable=new Runnable(){
            public void run() {
                Python py = Python.getInstance();
                py.getModule("main").callAttr("main");
            }
        };
        Thread thread=new Thread(runnable);
        thread.start();
        toast("Re-training Model");
    }

    public void sendWeights(View view){
//        /data/data/com.chaquo.python.console/files/chaquopy/AssetFinder/assets/local_model/mod1.npy
        String dir="chaquopy/AssetFinder/assets/local_model/mod1.npy";
        File file = new File(ConsoleActivity.this.getFilesDir(), dir);
        Runnable runnable=new Runnable(){
            public void run() {

                String deviceIP = getLocalIpAddress();
                Log.i("Info",deviceIP);

                HttpURLConnection conn = null;
                DataOutputStream os = null;
                DataInputStream inputStream = null;

                String urlServer = "http://"+ip+":8003/cmodel";

                String lineEnd = "\r\n";
                String twoHyphens = "--";
                String boundary =  "*****";
                int bytesRead, bytesAvailable, bufferSize, bytesUploaded = 0;
                byte[] buffer;
                int maxBufferSize = 2*1024*1024;

                String uploadname = "model.npy";

                try
                {
                    FileInputStream fis = new FileInputStream(file);

                    URL url = new URL(urlServer);
                    conn = (HttpURLConnection) url.openConnection();
                    conn.setChunkedStreamingMode(maxBufferSize);

                    // POST settings.
                    conn.setDoInput(true);
                    conn.setDoOutput(true);
                    conn.setUseCaches(false);
                    conn.setRequestMethod("POST");
                    conn.setRequestProperty("Connection", "Keep-Alive");
                    conn.setRequestProperty("Content-Type", "multipart/form-data; boundary="+boundary);
//                    conn.addRequestProperty("username", Username);
//                    conn.addRequestProperty("password", Password);
                    conn.connect();

                    os = new DataOutputStream(conn.getOutputStream());
                    os.writeBytes(twoHyphens + boundary + lineEnd);
                    os.writeBytes("Content-Disposition: form-data; name=\"uploadedfile\";filename=\"" + uploadname +"\"" + lineEnd);
                    os.writeBytes(lineEnd);

                    bytesAvailable = fis.available();
                    System.out.println("available: " + String.valueOf(bytesAvailable));
                    bufferSize = Math.min(bytesAvailable, maxBufferSize);
                    buffer = new byte[bufferSize];

                    bytesRead = fis.read(buffer, 0, bufferSize);
                    bytesUploaded += bytesRead;
                    while (bytesRead > 0)
                    {
                        os.write(buffer, 0, bufferSize);
                        bytesAvailable = fis.available();
                        bufferSize = Math.min(bytesAvailable, maxBufferSize);
                        buffer = new byte[bufferSize];
                        bytesRead = fis.read(buffer, 0, bufferSize);
                        bytesUploaded += bytesRead;
                    }
                    System.out.println("uploaded: "+String.valueOf(bytesUploaded));
                    os.writeBytes(lineEnd);
                    os.writeBytes(twoHyphens + boundary + twoHyphens + lineEnd);

                    // Responses from the server (code and message)
                    conn.setConnectTimeout(2000); // allow 2 seconds timeout.
                    int rcode = conn.getResponseCode();
                    if (rcode == 200) {
                        toast("Weights sent to Secure Aggregator");
                    }
                    else {
                        toast("Connection Failed");
                    }
                    fis.close();
                    os.flush();
                    os.close();
//                    Toast.makeText(getApplicationContext(), "Weights Sent", Toast.LENGTH_LONG).show();
                }
                catch (Exception ex)
                {
                    //ex.printStackTrace();
                    toast("Connection Failed");
                    //return false;
                }
            }
        };

        Thread thread= new Thread(runnable);
        thread.start();
    }

    public void sendSignal(View view) {
        Runnable runnable=new Runnable() {
            public void run() {
                try{
                    URL url = new URL("http://"+ip+":8000/clientstatus");
//                    toast(String.valueOf(url));
                    HttpURLConnection conn =(HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("POST");
                    conn.setRequestProperty("Content-Type","application/json; utf-8");
                    conn.setRequestProperty("Accept","application/json");
                    conn.setDoOutput(true);
                    String jsonInputString = String.format("{\"client_id\":\"%s\"}",getLocalIpAddress());
                    OutputStream os = conn.getOutputStream();
                    byte[] input = jsonInputString.getBytes(StandardCharsets.UTF_8);
                    os.write(input,0,input.length);
                    BufferedReader br = new BufferedReader(
                            new InputStreamReader(conn.getInputStream(),"utf-8"));
                    StringBuilder response = new StringBuilder();
                    String responseLine = null;
                    while((responseLine=br.readLine())!=null){
                        response.append(responseLine.trim());
                    }
                    toast(response.toString());
                    Log.i("msg0",response.toString());

                }catch(Exception e){
                    toast("Connection Failed");
                }
            }
        };
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Enter IP of Server");
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT );
        builder.setView(input);
//        builder.setView(R.layout.)
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                ip = input.getText().toString();
                Log.i("Info", ip);
                Thread thread = new Thread(runnable);
                thread.start();
//                new Thread(new ClientThread("message")).start();
            }
        });
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
            }
        });

        builder.show();
//        runOnUiThread(runnable);
//        thread.start();
    }

    public void downloadWeights(View view) {
//        toast("HI");
        Runnable runnable = new Runnable(){
            @Override
            public void run() {
                String path="http://"+ip+":8000/download";
                try {
                    URL url = new URL(path);

                    URLConnection ucon = url.openConnection();
                    ucon.setReadTimeout(5000);
                    ucon.setConnectTimeout(10000);

                    InputStream is = ucon.getInputStream();
                    BufferedInputStream inStream = new BufferedInputStream(is, 1024 * 5);

                    File file = new File(getApplicationContext().getFilesDir() + "/chaquopy/AssetFinder/assets/model_update/agg_model.h5");
//                    Log.i("Info",getApplicationContext().getFilesDir() + "/chaquopy/AssetFinder/assets/model_update/agg_model.h5");
                    if (file.exists()) {
                        file.delete();
                    }
                    file.createNewFile();

                    FileOutputStream outStream = new FileOutputStream(file);
                    byte[] buff = new byte[5 * 1024];

                    int len;
                    while ((len = inStream.read(buff)) != -1) {
                        outStream.write(buff, 0, len);
                    }

                    outStream.flush();
                    outStream.close();
                    inStream.close();
                    toast("Weights Received");

                } catch (Exception e) {
                    toast("Request Failed");
                    e.printStackTrace();
                }
            }
        };
        Thread thread=new Thread(runnable);
        thread.start();

    }

    public void expandView(View view){
        Animation animAppear = AnimationUtils.loadAnimation(getApplicationContext(),R.anim.appear);
        Animation animDisappear = AnimationUtils.loadAnimation(getApplicationContext(),R.anim.disappear);
        if(sendSignalButton.getVisibility()==View.VISIBLE){
            sendSignalButton.startAnimation(animDisappear);
            uploadButton.startAnimation(animDisappear);
            trainButton.startAnimation(animDisappear);
            downloadButton.startAnimation(animDisappear);
            sendSignalButton.setVisibility(View.INVISIBLE);
            uploadButton.setVisibility(View.INVISIBLE);
            trainButton.setVisibility(View.INVISIBLE);
            downloadButton.setVisibility(View.INVISIBLE);
            expandButton.setImageResource(R.drawable.grid_view_24px);
        }else {
            sendSignalButton.setVisibility(View.VISIBLE);
            uploadButton.setVisibility(View.VISIBLE);
            trainButton.setVisibility(View.VISIBLE);
            downloadButton.setVisibility(View.VISIBLE);
            sendSignalButton.startAnimation(animAppear);
            uploadButton.startAnimation(animAppear);
            trainButton.startAnimation(animAppear);
            downloadButton.startAnimation(animAppear);
            expandButton.setImageResource(R.drawable.arrow_back_ios_new_24px);
        }
    }

    enum Scroll {
        TOP, BOTTOM
    }
    private Scroll scrollRequest;
    
    public static class ConsoleModel extends ViewModel {
        boolean pendingNewline = false;  // Prevent empty line at bottom of screen
        int scrollChar = 0;              // Character offset of the top visible line.
        int scrollAdjust = 0;            // Pixels by which that line is scrolled above the top
                                         //   (prevents movement when keyboard hidden/shown).
    }
    private ConsoleModel consoleModel;

    protected Task task;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        consoleModel = ViewModelProviders.of(this).get(ConsoleModel.class);
        task = ViewModelProviders.of(this).get(getTaskClass());
        setContentView(resId("layout", "activity_console"));
        createInput();
        createOutput();
        sendSignalButton = (ImageButton) findViewById(R.id.send_signal_button);
        uploadButton = (ImageButton) findViewById(R.id.upload_button);
        trainButton = (ImageButton) findViewById(R.id.train_button);
        downloadButton = (ImageButton) findViewById(R.id.download_button);
        expandButton = (ImageButton) findViewById(R.id.expand_button);
        sendSignalButton.setVisibility(View.INVISIBLE);
        uploadButton.setVisibility(View.INVISIBLE);
        trainButton.setVisibility(View.INVISIBLE);
        downloadButton.setVisibility(View.INVISIBLE);
        StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
        StrictMode.setThreadPolicy(policy);
    }

    protected abstract Class<? extends Task> getTaskClass();

    private void createInput() {
        etInput = findViewById(resId("id", "etInput"));

        // Strip formatting from pasted text.
        etInput.addTextChangedListener(new TextWatcher() {
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {}
            public void onTextChanged(CharSequence s, int start, int before, int count) {}
            public void afterTextChanged(Editable e) {
                for (CharacterStyle cs : e.getSpans(0, e.length(), CharacterStyle.class)) {
                    e.removeSpan(cs);
                }
            }
        });

        // At least on API level 28, if an ACTION_UP is lost during a rotation, then the app
        // (or any other app which takes focus) will receive an endless stream of ACTION_DOWNs
        // until the key is pressed again. So we react to ACTION_UP instead.
        etInput.setOnEditorActionListener(new TextView.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                if (actionId == EditorInfo.IME_ACTION_DONE ||
                    (event != null && event.getAction() == KeyEvent.ACTION_UP)) {
                    String text = etInput.getText().toString() + "\n";
                    etInput.setText("");
                    output(span(text, new StyleSpan(Typeface.BOLD)));
                    scrollTo(Scroll.BOTTOM);
                    task.onInput(text);
                }

                // If we return false on ACTION_DOWN, we won't be given the ACTION_UP.
                return true;
            }
        });

        task.inputEnabled.observe(this, new Observer<Boolean>() {
            @Override public void onChanged(@Nullable Boolean enabled) {
                InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
                if (enabled) {
                    etInput.setVisibility(View.VISIBLE);
                    etInput.setEnabled(true);

                    // requestFocus alone doesn't always bring up the soft keyboard during startup
                    // on the Nexus 4 with API level 22: probably some race condition. (After
                    // rotation with input *already* enabled, the focus may be overridden by
                    // onRestoreInstanceState, which will run after this observer.)
                    etInput.requestFocus();
                    imm.showSoftInput(etInput, InputMethodManager.SHOW_IMPLICIT);
                } else {
                    // Disable rather than hide, otherwise tvOutput gets a gray background on API
                    // level 26, like tvCaption in the main menu when you press an arrow key.
                    etInput.setEnabled(false);
                    imm.hideSoftInputFromWindow(tvOutput.getWindowToken(), 0);
                }
            }
        });
    }

    private void createOutput() {
        svOutput = findViewById(resId("id", "svOutput"));
        svOutput.getViewTreeObserver().addOnGlobalLayoutListener(this);

        tvOutput = findViewById(resId("id", "tvOutput"));
        if (Build.VERSION.SDK_INT >= 23) {
            // noinspection WrongConstant
            tvOutput.setBreakStrategy(Layout.BREAK_STRATEGY_SIMPLE);
        }
        // Don't start observing task.output yet: we need to restore the scroll position first so
        // we maintain the scrolled-to-bottom state.
    }

    @Override protected void onRestoreInstanceState(@NonNull Bundle savedInstanceState) {
        // Don't restore the UI state unless we have the non-UI state as well.
        if (task.getState() != Thread.State.NEW) {
            super.onRestoreInstanceState(savedInstanceState);
        }
    }

    @Override protected void onResume() {
        super.onResume();
        // Needs to be in onResume rather than onStart because onRestoreInstanceState runs
        // between them.
        if (task.getState() == Thread.State.NEW) {
            task.start();
        }
    }

    @Override protected void onPause() {
        super.onPause();
        saveScroll();  // Necessary to save bottom position in case we've never scrolled.
    }

    // This callback is run after onResume, after each layout pass. If a view's size, position
    // or visibility has changed, the new values will be visible here.
    @Override public void onGlobalLayout() {
        if (outputWidth != svOutput.getWidth() || outputHeight != svOutput.getHeight()) {
            // Can't register this listener in onCreate on API level 15
            // (https://stackoverflow.com/a/35054919).
            if (outputWidth == -1) {
                svOutput.getViewTreeObserver().addOnScrollChangedListener(this);
            }

            // Either we've just started up, or the keyboard has been hidden or shown.
            outputWidth = svOutput.getWidth();
            outputHeight = svOutput.getHeight();
            restoreScroll();
        } else if (scrollRequest != null) {
            int y = -1;
            switch (scrollRequest) {
                case TOP:
                    y = 0;
                    break;
                case BOTTOM:
                    y = tvOutput.getHeight();
                    break;
            }

            // Don't use smooth scroll, because if an output call happens while it's animating
            // towards the bottom, isScrolledToBottom will believe we've left the bottom and
            // auto-scrolling will stop. Don't use fullScroll either, because not only does it use
            // smooth scroll, it also grabs focus.
            svOutput.scrollTo(0, y);
            scrollRequest = null;
        }
    }

   @Override public void onScrollChanged() {
        saveScroll();
    }

    // After a rotation, a ScrollView will restore the previous pixel scroll position. However, due
    // to re-wrapping, this may result in a completely different piece of text being visible. We'll
    // try to maintain the text position of the top line, unless the view is scrolled to the bottom,
    // in which case we'll maintain that. Maintaining the bottom line will also cause a scroll
    // adjustment when the keyboard's hidden or shown.
    private void saveScroll() {
        if (isScrolledToBottom()) {
            consoleModel.scrollChar = tvOutput.getText().length();
            consoleModel.scrollAdjust = 0;
        } else {
            int scrollY = svOutput.getScrollY();
            Layout layout = tvOutput.getLayout();
            if (layout != null) {  // See note in restoreScroll
                int line = layout.getLineForVertical(scrollY);
                consoleModel.scrollChar = layout.getLineStart(line);
                consoleModel.scrollAdjust = scrollY - layout.getLineTop(line);
            }
        }
    }

    private void restoreScroll() {
        removeCursor();

        // getLayout sometimes returns null even when called from onGlobalLayout. The
        // documentation says this can happen if the "text or width has recently changed", but
        // does not define "recently". See Electron Cash issues #1330 and #1592.
        Layout layout = tvOutput.getLayout();
        if (layout != null) {
            int line = layout.getLineForOffset(consoleModel.scrollChar);
            svOutput.scrollTo(0, layout.getLineTop(line) + consoleModel.scrollAdjust);
        }

        // If we are now scrolled to the bottom, we should stick there. (scrollTo probably won't
        // trigger onScrollChanged unless the scroll actually changed.)
        saveScroll();

        task.output.removeObservers(this);
        task.output.observe(this, new Observer<CharSequence>() {
            @Override public void onChanged(@Nullable CharSequence text) {
                output(text);
            }
        });
    }

    private boolean isScrolledToBottom() {
        int visibleHeight = (svOutput.getHeight() - svOutput.getPaddingTop() -
                             svOutput.getPaddingBottom());
        int maxScroll = Math.max(0, tvOutput.getHeight() - visibleHeight);
        return (svOutput.getScrollY() >= maxScroll);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater mi = getMenuInflater();
        mi.inflate(resId("menu", "top_bottom"), menu);
        return true;
    }

    @Override public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == resId("id", "menu_top")) {
            scrollTo(Scroll.TOP);
        } else if (id == resId("id", "menu_bottom")) {
            scrollTo(Scroll.BOTTOM);
        } else {
            return false;
        }
        return true;
    }

    public static Spannable span(CharSequence text, Object... spans) {
        Spannable spanText = new SpannableStringBuilder(text);
        for (Object span : spans) {
            spanText.setSpan(span, 0, text.length(), 0);
        }
        return spanText;
    }

    private void output(CharSequence text) {
        removeCursor();
        if (consoleModel.pendingNewline) {
            tvOutput.append("\n");
            consoleModel.pendingNewline = false;
        }
        if (text.charAt(text.length() - 1) == '\n') {
            tvOutput.append(text.subSequence(0, text.length() - 1));
            consoleModel.pendingNewline = true;
        } else {
            tvOutput.append(text);
        }

        Editable scrollback = (Editable) tvOutput.getText();
        if (scrollback.length() > MAX_SCROLLBACK_LEN) {
            scrollback.delete(0, MAX_SCROLLBACK_LEN / 10);
        }

        // Changes to the TextView height won't be reflected by getHeight until after the
        // next layout pass, so isScrolledToBottom is safe here.
        if (isScrolledToBottom()) {
            scrollTo(Scroll.BOTTOM);
        }
    }

    // Don't actually scroll until the next onGlobalLayout, when we'll know what the new TextView
    // height is.
    private void scrollTo(Scroll request) {
        // The "top" button should take priority over an auto-scroll.
        if (scrollRequest != Scroll.TOP) {
            scrollRequest = request;
            svOutput.requestLayout();
        }
    }

    // Because we've set textIsSelectable, the TextView will create an invisible cursor (i.e. a
    // zero-length selection) during startup, and re-create it if necessary whenever the user taps
    // on the view. When a TextView is focused and it has a cursor, it will adjust its containing
    // ScrollView whenever the text changes in an attempt to keep the cursor on-screen.
    // textIsSelectable implies focusable, so if there are no other focusable views in the layout,
    // then it will always be focused.
    //
    // To avoid interference from this, we'll remove any cursor before we adjust the scroll.
    // A non-zero-length selection is left untouched and may affect the scroll in the normal way,
    // which is fine because it'll only exist if the user deliberately created it.
    private void removeCursor() {
        Spannable text = (Spannable) tvOutput.getText();
        int selStart = Selection.getSelectionStart(text);
        int selEnd = Selection.getSelectionEnd(text);

        // When textIsSelectable is set, the buffer type after onRestoreInstanceState is always
        // Spannable, regardless of the value of bufferType. It would then become Editable (and
        // have a cursor added), during the first call to append(). Make that happen now so we can
        // remove the cursor before append() is called.
        if (!(text instanceof Editable)) {
            tvOutput.setText(text, TextView.BufferType.EDITABLE);
            text = (Editable) tvOutput.getText();

            // setText removes any existing selection, at least on API level 26.
            if (selStart >= 0) {
                Selection.setSelection(text, selStart, selEnd);
            }
        }

        if (selStart >= 0 && selStart == selEnd) {
            Selection.removeSelection(text);
        }

    }

    public int resId(String type, String name) {
        return Utils.resId(this, type, name);
    }

    // =============================================================================================

    public static abstract class Task extends AndroidViewModel {

        private Thread.State state = Thread.State.NEW;

        public void start() {
            new Thread(() -> {
                try {
                    Task.this.run();
                    output(spanColor("[Finished]", resId("color", "console_meta")));
                } finally {
                    inputEnabled.postValue(false);
                    state = Thread.State.TERMINATED;
                }
            }).start();
            state = Thread.State.RUNNABLE;
        }

        public Thread.State getState() { return state; }

        public MutableLiveData<Boolean> inputEnabled = new MutableLiveData<>();
        public BufferedLiveEvent<CharSequence> output = new BufferedLiveEvent<>();

        public Task(Application app) {
            super(app);
            inputEnabled.setValue(false);
        }

        /** Override this method to provide the task's implementation. It will be called on a
         *  background thread. */
        public abstract void run();

        /** Called on the UI thread each time the user enters some input, A trailing newline is
         * always included. The base class implementation does nothing. */
        public void onInput(String text) {}

        public void output(final CharSequence text) {
            if (text.length() == 0) return;
            output.postValue(text);
//            output.postValue(span(text, new ForegroundColorSpan(Color.GREEN)));
        }

        public void outputError(CharSequence text) {
            output(spanColor(text, resId("color", "console_error")));
        }

        public Spannable spanColor(CharSequence text, int colorId) {
            int color = ContextCompat.getColor(this.getApplication(), colorId);
            return span(text, new ForegroundColorSpan(color));
        }

        public int resId(String type, String name) {
            return Utils.resId(getApplication(), type, name);
        }
    }

}
