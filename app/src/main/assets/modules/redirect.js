let map = {
    BuildConfig: org.autojs.autojs6.BuildConfig,
    R: org.autojs.autojs6.R,
    app: {
        AppOpsKt: org.autojs.autojs.app.AppOps,
        DialogUtils: org.autojs.autojs.app.DialogUtils,
        FragmentPagerAdapterBuilder: org.autojs.autojs.app.FragmentPagerAdapterBuilder,
        GlobalAppContext: org.autojs.autojs.app.GlobalAppContext,
        OnActivityResultDelegate: org.autojs.autojs.app.OnActivityResultDelegate,
        OperationDialogBuilder: org.autojs.autojs.app.CircularMenuOperationDialogBuilder,
        SimpleActivityLifecycleCallbacks: org.autojs.autojs.app.SimpleActivityLifecycleCallbacks,
        isUsageStatsPermissionGranted: org.autojs.autojs.app.AppOps.isUsageStatsPermissionGranted,
    },
    autojs: {
        BuildConfig: org.autojs.autojs6.BuildConfig,
        R: org.autojs.autojs6.R,
        ScriptEngineService: org.autojs.autojs.engine.ScriptEngineService,
        annotation: {
            ScriptClass: org.autojs.autojs.annotation.ScriptClass,
            ScriptInterface: org.autojs.autojs.annotation.ScriptInterface,
            ScriptVariable: org.autojs.autojs.annotation.ScriptVariable,
        },
        codegeneration: {
            CodeGenerator: org.autojs.autojs.codegeneration.CodeGenerator,
        },
        core: {
            accessibility: {
                AccessibilityBridge: org.autojs.autojs.core.accessibility.AccessibilityBridge,
                AccessibilityServiceTool: org.autojs.autojs.core.accessibility.AccessibilityServiceTool,
                SimpleActionAutomator: org.autojs.autojs.core.accessibility.SimpleActionAutomator,
                UiSelector: org.autojs.autojs.core.accessibility.UiSelector,
            },
            activity: {
                ActivityInfoProvider: org.autojs.autojs.core.activity.ActivityInfoProvider,
            },
            broadcast: {
                BroadcastEmitter: org.autojs.autojs.core.broadcast.BroadcastEmitter,
            },
            console: {
                ConsoleImpl: org.autojs.autojs.core.console.ConsoleImpl,
                ConsoleView: org.autojs.autojs.core.console.ConsoleView,
                GlobalConsole: org.autojs.autojs.core.console.GlobalConsole,
            },
            eventloop: {
                EventEmitter: org.autojs.autojs.core.eventloop.EventEmitter,
                SimpleEvent: org.autojs.autojs.core.eventloop.SimpleEvent,
            },
            floaty: {
                BaseResizableFloatyWindow: org.autojs.autojs.core.floaty.BaseResizableFloatyWindow,
                RawWindow: org.autojs.autojs.core.floaty.RawWindow,
            },
            graphics: {
                ScriptCanvasView: org.autojs.autojs.core.graphics.ScriptCanvasView,
            },
            image: {
                ColorFinder: org.autojs.autojs.core.image.ColorFinder,
                Colors: org.autojs.autojs.core.image.Colors,
                ImageWrapper: org.autojs.autojs.core.image.ImageWrapper,
                TemplateMatching: org.autojs.autojs.core.image.TemplateMatching,
                capture: {
                    ScreenCaptureRequestActivity: org.autojs.autojs.core.image.capture.ScreenCaptureRequestActivity,
                    ScreenCaptureRequester: org.autojs.autojs.core.image.capture.ScreenCaptureRequester,
                    ScreenCapturer: org.autojs.autojs.core.image.capture.ScreenCapturer,
                    ScreenCapturerForegroundService: org.autojs.autojs.core.image.capture.ScreenCapturerForegroundService,
                },
            },
            inputevent: {
                InputDevices: org.autojs.autojs.core.inputevent.InputDevices,
                InputEventCodes: org.autojs.autojs.core.inputevent.InputEventCodes,
                InputEventObserver: org.autojs.autojs.core.inputevent.InputEventObserver,
                RootAutomator: org.autojs.autojs.core.inputevent.RootAutomator,
                ShellKeyObserver: org.autojs.autojs.core.inputevent.ShellKeyObserver,
                TouchObserver: org.autojs.autojs.core.inputevent.TouchObserver,
            },
            internal: {
                Functions: org.autojs.autojs.core.internal.Functions,
            },
            looper: {
                LooperHelper: org.autojs.autojs.core.looper.LooperHelper,
                Loopers: org.autojs.autojs.core.looper.Loopers,
                MainThreadProxy: org.autojs.autojs.core.looper.MainThreadProxy,
                Timer: org.autojs.autojs.core.looper.Timer,
                TimerThread: org.autojs.autojs.core.looper.TimerThread,
            },
            opencv: {
                Mat: org.autojs.autojs.core.opencv.Mat,
                MatOfPoint: org.autojs.autojs.core.opencv.MatOfPoint,
                OpenCVHelper: org.autojs.autojs.core.opencv.OpenCVHelper,
            },
            permission: {
                OnRequestPermissionsResultCallback: org.autojs.autojs.core.permission.OnRequestPermissionsResultCallback,
                PermissionRequestProxyActivity: org.autojs.autojs.core.permission.PermissionRequestProxyActivity,
                Permissions: org.autojs.autojs.permission.Base,
                RequestPermissionCallbacks: org.autojs.autojs.core.permission.RequestPermissionCallbacks,
            },
            plugin: {
                Plugin: org.autojs.autojs.core.plugin.Plugin,
            },
            pref: {
                Pref: org.autojs.autojs.pref.Pref,
            },
            record: {
                Recorder: org.autojs.autojs.core.record.Recorder,
                accessibility: {
                    AccessibilityActionRecorder: org.autojs.autojs.core.record.accessibility.AccessibilityActionRecorder,
                },
                inputevent: {
                    EventFormatException: org.autojs.autojs.core.record.inputevent.EventFormatException,
                    InputEventRecorder: org.autojs.autojs.core.record.inputevent.InputEventRecorder,
                    InputEventToAutoFileRecorder: org.autojs.autojs.core.record.inputevent.InputEventToAutoFileRecorder,
                    InputEventToRootAutomatorRecorder: org.autojs.autojs.core.record.inputevent.InputEventToRootAutomatorRecorder,
                    TouchRecorder: org.autojs.autojs.core.record.inputevent.TouchRecorder,
                },
            },
            ui: {
                BaseEvent: org.autojs.autojs.core.ui.BaseEvent,
                JsViewHelper: org.autojs.autojs.core.ui.JsViewHelper,
                ViewExtras: org.autojs.autojs.core.ui.ViewExtras,
                attribute: {
                    ViewAttributes: org.autojs.autojs.core.ui.attribute.ViewAttributes,
                    ViewAttributesFactory: org.autojs.autojs.core.ui.attribute.ViewAttributesFactory,
                },
                dialog: {
                    BlockedMaterialDialog: org.autojs.autojs.core.ui.dialog.BlockedMaterialDialog,
                    JsDialogBuilder: org.autojs.autojs.core.ui.dialog.JsDialogBuilder,
                },
                inflater: {
                    DynamicLayoutInflater: org.autojs.autojs.core.ui.inflater.DynamicLayoutInflater,
                    ImageLoader: org.autojs.autojs.core.ui.inflater.ImageLoader,
                    ResourceParser: org.autojs.autojs.core.ui.inflater.ResourceParser,
                    ShouldCallOnFinishInflate: org.autojs.autojs.core.ui.inflater.ShouldCallOnFinishInflate,
                    ViewCreator: org.autojs.autojs.core.ui.inflater.ViewCreator,
                    ViewInflater: org.autojs.autojs.core.ui.inflater.ViewInflater,
                    inflaters: {
                        AppBarInflater: org.autojs.autojs.core.ui.inflater.inflaters.AppBarInflater,
                        BaseViewInflater: org.autojs.autojs.core.ui.inflater.inflaters.BaseViewInflater,
                        CanvasViewInflater: org.autojs.autojs.core.ui.inflater.inflaters.CanvasViewInflater,
                        DatePickerInflater: org.autojs.autojs.core.ui.inflater.inflaters.DatePickerInflater,
                        Exceptions: org.autojs.autojs.core.ui.inflater.inflaters.Exceptions,
                        FrameLayoutInflater: org.autojs.autojs.core.ui.inflater.inflaters.FrameLayoutInflater,
                        ImageViewInflater: org.autojs.autojs.core.ui.inflater.inflaters.ImageViewInflater,
                        JsGridViewInflater: org.autojs.autojs.core.ui.inflater.inflaters.JsGridViewInflater,
                        JsImageViewInflater: org.autojs.autojs.core.ui.inflater.inflaters.JsImageViewInflater,
                        JsListViewInflater: org.autojs.autojs.core.ui.inflater.inflaters.JsListViewInflater,
                        LinearLayoutInflater: org.autojs.autojs.core.ui.inflater.inflaters.LinearLayoutInflater,
                        ProgressBarInflater: org.autojs.autojs.core.ui.inflater.inflaters.ProgressBarInflater,
                        RadioGroupInflater: org.autojs.autojs.core.ui.inflater.inflaters.RadioGroupInflater,
                        SpinnerInflater: org.autojs.autojs.core.ui.inflater.inflaters.SpinnerInflater,
                        TabLayoutInflater: org.autojs.autojs.core.ui.inflater.inflaters.TabLayoutInflater,
                        TextViewInflater: org.autojs.autojs.core.ui.inflater.inflaters.TextViewInflater,
                        TimePickerInflater: org.autojs.autojs.core.ui.inflater.inflaters.TimePickerInflater,
                        ToolbarInflater: org.autojs.autojs.core.ui.inflater.inflaters.ToolbarInflater,
                        ViewGroupInflater: org.autojs.autojs.core.ui.inflater.inflaters.ViewGroupInflater,
                    },
                    util: {
                        Colors: org.autojs.autojs.core.ui.inflater.util.Colors,
                        Dimensions: org.autojs.autojs.core.ui.inflater.util.Dimensions,
                        Drawables: org.autojs.autojs.core.ui.inflater.util.Drawables,
                        Gravities: org.autojs.autojs.core.ui.inflater.util.Gravities,
                        Ids: org.autojs.autojs.core.ui.inflater.util.Ids,
                        Res: org.autojs.autojs.core.ui.inflater.util.Res,
                        Strings: org.autojs.autojs.core.ui.inflater.util.Strings,
                        ValueMapper: org.autojs.autojs.core.ui.inflater.util.ValueMapper,
                    },
                },
                nativeview: {
                    NativeView: org.autojs.autojs.core.ui.nativeview.NativeView,
                    ViewPrototype: org.autojs.autojs.core.ui.nativeview.ViewPrototype,
                },
                widget: {
                    CustomSnackbar: org.autojs.autojs.core.ui.widget.CustomSnackbar,
                    JsButton: org.autojs.autojs.core.ui.widget.JsButton,
                    JsEditText: org.autojs.autojs.core.ui.widget.JsEditText,
                    JsFrameLayout: org.autojs.autojs.core.ui.widget.JsFrameLayout,
                    JsGridView: org.autojs.autojs.core.ui.widget.JsGridView,
                    JsImageView: org.autojs.autojs.core.ui.widget.JsImageView,
                    JsLinearLayout: org.autojs.autojs.core.ui.widget.JsLinearLayout,
                    JsListView: org.autojs.autojs.core.ui.widget.JsListView,
                    JsRelativeLayout: org.autojs.autojs.core.ui.widget.JsRelativeLayout,
                    JsSpinner: org.autojs.autojs.core.ui.widget.JsSpinner,
                    JsTabLayout: org.autojs.autojs.core.ui.widget.JsTabLayout,
                    JsTextView: org.autojs.autojs.core.ui.widget.JsTextView,
                    JsTextViewLegacy: org.autojs.autojs.core.ui.widget.JsTextViewLegacy,
                    JsToolbar: org.autojs.autojs.core.ui.widget.JsToolbar,
                    JsViewPager: org.autojs.autojs.core.ui.widget.JsViewPager,
                    JsWebView: org.autojs.autojs.core.ui.widget.JsWebView,
                },
                xml: {
                    XmlConverter: org.autojs.autojs.core.ui.xml.XmlConverter,
                },
            },
            util: {
                ProcessShell: org.autojs.autojs.runtime.api.ProcessShell,
                ScriptPromiseAdapter: org.autojs.autojs.runtime.api.ScriptPromiseAdapter,
                Shell: org.autojs.autojs.runtime.api.Shell,
            },
        },
        engine: {
            JavaScriptEngine: org.autojs.autojs.engine.JavaScriptEngine,
            LoopBasedJavaScriptEngine: org.autojs.autojs.engine.LoopBasedJavaScriptEngine,
            RhinoJavaScriptEngine: org.autojs.autojs.engine.RhinoJavaScriptEngine,
            RootAutomatorEngine: org.autojs.autojs.engine.RootAutomatorEngine,
            ScriptEngine: org.autojs.autojs.engine.ScriptEngine,
            ScriptEngineManager: org.autojs.autojs.engine.ScriptEngineManager,
            encryption: {
                ScriptEncryption: org.autojs.autojs.engine.encryption.ScriptEncryption,
            },
            module: {
                AssetAndUrlModuleSourceProvider: org.autojs.autojs.engine.module.AssetAndUrlModuleSourceProvider,
            },
            preprocess: {
                MultiLinePreprocessor: org.autojs.autojs.engine.preprocess.MultiLinePreprocessor,
            },
        },
        execution: {
            ExecutionConfig: org.autojs.autojs.execution.ExecutionConfig,
            LoopedBasedJavaScriptExecution: org.autojs.autojs.execution.LoopedBasedJavaScriptExecution,
            RunnableScriptExecution: org.autojs.autojs.execution.RunnableScriptExecution,
            ScriptExecuteActivity: org.autojs.autojs.execution.ScriptExecuteActivity,
            ScriptExecution: org.autojs.autojs.execution.ScriptExecution,
            ScriptExecutionListener: org.autojs.autojs.execution.ScriptExecutionListener,
            ScriptExecutionObserver: org.autojs.autojs.execution.ScriptExecutionObserver,
            ScriptExecutionTask: org.autojs.autojs.execution.ScriptExecutionTask,
            SimpleScriptExecutionListener: org.autojs.autojs.execution.SimpleScriptExecutionListener,
        },
        // inrt: {
        //     BuildConfig: org.autojs.autojs.inrt.BuildConfig,
        //     LogActivity: org.autojs.autojs.inrt.LogActivity,
        //     Pref: org.autojs.autojs.inrt.Pref,
        //     R: org.autojs.autojs.inrt.R,
        //     SettingsActivity: org.autojs.autojs.inrt.SettingsActivity,
        //     autojs: {
        //         AutoJs: org.autojs.autojs.inrt.autojs.AutoJs,
        //         GlobalKeyObserver: org.autojs.autojs.inrt.autojs.GlobalKeyObserver,
        //     },
        //     launch: {
        //         GlobalProjectLauncher: org.autojs.autojs.inrt.launch.GlobalProjectLauncher,
        //     },
        // },
        project: {
            BuildInfo: org.autojs.autojs.project.BuildInfo,
            ProjectConfig: org.autojs.autojs.project.ProjectConfig,
            ProjectLauncher: org.autojs.autojs.project.ProjectLauncher,
            ScriptConfig: org.autojs.autojs.project.ScriptConfig,
        },
        rhino: {
            AndroidClassLoader: org.autojs.autojs.rhino.AndroidClassLoader,
            AutoJsContext: org.autojs.autojs.rhino.AutoJsContext,
            InterruptibleAndroidContextFactory: org.autojs.autojs.rhino.InterruptibleAndroidContextFactory,
            NativeJavaObjectWithPrototype: org.autojs.autojs.rhino.NativeJavaObjectWithPrototype,
            ProxyObject: org.autojs.autojs.rhino.ProxyObject,
            RhinoAndroidHelper: org.autojs.autojs.rhino.RhinoAndroidHelper,
            TokenStream: org.autojs.autojs.rhino.TokenStream,
            TopLevelScope: org.autojs.autojs.rhino.TopLevelScope,
            continuation: {
                Continuation: org.autojs.autojs.rhino.continuation.Continuation,
            },
            debug: {
                DebugCallback: org.autojs.autojs.rhino.debug.DebugCallback,
                Debugger: org.autojs.autojs.rhino.debug.Debugger,
                Dim: org.autojs.autojs.rhino.debug.Dim,
            },
        },
        runtime: {
            ScriptBridges: org.autojs.autojs.runtime.ScriptBridges,
            ScriptRuntime: org.autojs.autojs.runtime.ScriptRuntime,
            accessibility: {
                AccessibilityConfig: org.autojs.autojs.runtime.accessibility.AccessibilityConfig,
            },
            api: {
                AbstractConsole: org.autojs.autojs.runtime.api.AbstractConsole,
                AbstractShell: org.autojs.autojs.runtime.api.AbstractShell,
                AppUtils: org.autojs.autojs.runtime.api.AppUtils,
                Console: org.autojs.autojs.runtime.api.Console,
                Device: org.autojs.autojs.runtime.api.Device,
                Dialogs: org.autojs.autojs.runtime.api.Dialogs,
                Engines: org.autojs.autojs.runtime.api.Engines,
                Events: org.autojs.autojs.runtime.api.Events,
                Files: org.autojs.autojs.runtime.api.Files,
                Floaty: org.autojs.autojs.runtime.api.Floaty,
                Images: org.autojs.autojs.runtime.api.Images,
                Media: org.autojs.autojs.runtime.api.Media,
                Plugins: org.autojs.autojs.runtime.api.Plugins,
                Sensors: org.autojs.autojs.runtime.api.Sensors,
                Threads: org.autojs.autojs.runtime.api.Threads,
                Timers: org.autojs.autojs.runtime.api.Timers,
                UI: org.autojs.autojs.runtime.api.UI,
            },
            exception: {
                ScriptEnvironmentException: org.autojs.autojs.runtime.exception.ScriptEnvironmentException,
                ScriptException: org.autojs.autojs.runtime.exception.ScriptException,
                ScriptInterruptedException: org.autojs.autojs.runtime.exception.ScriptInterruptedException,
            },
        },
        script: {
            AutoFileSource: org.autojs.autojs.script.AutoFileSource,
            EncryptedScriptFileHeader: org.autojs.autojs.script.EncryptedScriptFileHeader,
            JavaScriptFileSource: org.autojs.autojs.script.JavaScriptFileSource,
            JavaScriptSource: org.autojs.autojs.script.JavaScriptSource,
            JsBeautifier: org.autojs.autojs.script.JsBeautifier,
            ScriptSource: org.autojs.autojs.script.ScriptSource,
            SequenceScriptSource: org.autojs.autojs.script.SequenceScriptSource,
            StringScriptSource: org.autojs.autojs.script.StringScriptSource,
        },
        util: {
            FloatingPermission: {
                ensurePermissionGranted(context) {
                    let oo = new org.autojs.autojs.permission.DisplayOverOtherAppsPermission(context);
                    return Boolean(oo.has() || oo.request());
                },
            },
            ForegroundServiceCreator: org.autojs.autojs.tool.ForegroundServiceCreator,
            ForegroundServiceUtils: org.autojs.autojs.util.ForegroundServiceUtils,
            ProcessUtils: org.autojs.autojs.util.ProcessUtils,
        },
        workground: {
            WrapContentLinearLayoutManager: org.autojs.autojs.groundwork.WrapContentLinearLayoutManager,
        },
    },
    automator: {
        ActionArgument: org.autojs.autojs.core.automator.ActionArgument,
        BuildConfig: org.autojs.autojs6.BuildConfig,
        GlobalActionAutomator: org.autojs.autojs.core.automator.GlobalActionAutomator,
        UiGlobalSelector: org.autojs.autojs.core.accessibility.UiSelector,
        UiObject: org.autojs.autojs.core.automator.UiObject,
        UiObjectCollection: org.autojs.autojs.core.automator.UiObjectCollection,
        filter: {
            Filter: org.autojs.autojs.core.automator.filter.Filter,
        },
        search: {
            BFS: org.autojs.autojs.core.automator.search.BFS,
            DFS: org.autojs.autojs.core.automator.search.DFS,
            SearchAlgorithm: org.autojs.autojs.core.automator.search.SearchAlgorithm,
        },
        simple_action: {
            ActionFactory: org.autojs.autojs.core.automator.action.ActionFactory,
            ActionTarget: org.autojs.autojs.core.automator.action.ActionTarget,
            FilterAction: org.autojs.autojs.core.automator.action.FilterAction,
            SimpleAction: org.autojs.autojs.core.automator.action.SimpleAction,
        },
        test: {
            TestUiObject: org.autojs.autojs.core.automator.test.TestUiObject,
        },
    },
    concurrent: {
        VolatileBox: org.autojs.autojs.concurrent.VolatileBox,
        VolatileDispose: org.autojs.autojs.concurrent.VolatileDispose,
    },
    event: {
        EventDispatcher: org.autojs.autojs.event.EventDispatcher,
    },
    ext: {
        ifNull: org.autojs.autojs.util.KotlinUtils.ifNull,
    },
    io: {
        ConcatReader: org.autojs.autojs.io.ConcatReader,
        Zip: org.autojs.autojs.io.Zip,
    },
    lang: {
        ThreadCompat: org.autojs.autojs.lang.ThreadCompat,
    },
    notification: {
        Notification: org.autojs.autojs.core.notification.Notification,
        NotificationListenerService: org.autojs.autojs.core.notification.NotificationListenerService,
    },
    pio: {
        PFile: org.autojs.autojs.pio.PFile,
        PFileInterface: org.autojs.autojs.pio.PFileInterface,
        PFiles: org.autojs.autojs.pio.PFiles,
        UncheckedIOException: org.autojs.autojs.pio.UncheckedIOException,
    },
    theme: {
        ThemeColor: org.autojs.autojs.theme.ThemeColor,
        ThemeColorHelper: org.autojs.autojs.theme.ThemeColorHelper,
        ThemeColorManager: org.autojs.autojs.theme.ThemeColorManager,
        ThemeColorMutable: org.autojs.autojs.theme.ThemeColorMutable,
        app: {
            ColorSelectActivity: org.autojs.autojs.theme.app.ColorSelectActivity,
        },
        internal: {
            DrawableTool: org.autojs.autojs.util.DrawableUtils,
            ScrollingViewEdgeGlowColorHelper: org.autojs.autojs.theme.internal.ScrollingViewEdgeGlowColorHelper,
        },
        preference: {
            ThemeColorPreferenceFragment: org.autojs.autojs.theme.preference.ThemeColorPreferenceFragment,
        },
        util: {
            ListBuilder: org.autojs.autojs.theme.util.ListBuilder,
        },
    },
    util: {
        AdvancedEncryptionStandard: org.autojs.autojs.engine.encryption.AdvancedEncryptionStandard,
        ArrayUtils: org.autojs.autojs.util.ArrayUtils,
        BackPressedHandler: org.autojs.autojs.event.BackPressedHandler,
        BiMap: org.autojs.autojs.core.ui.BiMap,
        BiMaps: org.autojs.autojs.core.ui.BiMaps,
        Callback: org.autojs.autojs.tool.Callback,
        ClipboardUtil: org.autojs.autojs.util.ClipboardUtils,
        Consumer: org.autojs.autojs.tool.Consumer,
        DeveloperUtils: org.autojs.autojs.util.DeveloperUtils,
        DrawerAutoClose: org.autojs.autojs.ui.widget.DrawerAutoClose,
        Func1: org.autojs.autojs.tool.Func1,
        HashUtils: org.autojs.autojs.util.MD5Utils,
        IntentExtras: org.autojs.autojs.tool.IntentExtras,
        IntentUtil: org.autojs.autojs.util.IntentUtils,
        MD5: org.autojs.autojs.util.MD5Utils,
        MapBuilder: org.autojs.autojs.tool.MapBuilder,
        MimeTypes: org.autojs.autojs.util.MimeTypesUtils,
        Nath: org.autojs.autojs.util.MathUtils,
        Objects: org.autojs.autojs.util.Objects,
        ScreenMetrics: org.autojs.autojs.runtime.api.ScreenMetrics,
        SdkVersionUtil: org.autojs.autojs.util.SdkVersionUtils,
        SparseArrayEntries: org.autojs.autojs.tool.SparseArrayEntries,
        Supplier: org.autojs.autojs.tool.Supplier,
        TextUtils: org.autojs.autojs.util.StringUtils,
        UiHandler: org.autojs.autojs.tool.UiHandler,
        ViewUtil: org.autojs.autojs.util.ViewUtils,
        ViewUtils: org.autojs.autojs.util.ViewUtils,
        sortedArrayOf: org.autojs.autojs.util.KotlinUtils.sortedArrayOf,
    },
    view: {
        accessibility: {
            AccessibilityDelegate: org.autojs.autojs.core.accessibility.AccessibilityDelegate,
            AccessibilityNodeInfoAllocator: org.autojs.autojs.core.accessibility.AccessibilityNodeInfoAllocator,
            AccessibilityNodeInfoHelper: org.autojs.autojs.core.accessibility.AccessibilityNodeInfoHelper,
            AccessibilityNotificationObserver: org.autojs.autojs.core.accessibility.AccessibilityNotificationObserver,
            AccessibilityService: org.autojs.autojs.core.accessibility.AccessibilityService,
            KeyInterceptor: org.autojs.autojs.core.accessibility.KeyInterceptor,
            LayoutInspector: org.autojs.autojs.core.accessibility.LayoutInspector,
            NodeInfo: org.autojs.autojs.core.accessibility.NodeInfo,
            NotificationListener: org.autojs.autojs.core.accessibility.NotificationListener,
            OnKeyListener: org.autojs.autojs.core.accessibility.OnKeyListener,
        },
    },
};

module.exports = Object.assign(com.stardust, map);
